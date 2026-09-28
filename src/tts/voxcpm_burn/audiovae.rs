use std::{marker::PhantomData, path::Path};

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1dConfig, ConvTranspose1dConfig},
        Tanh,
    },
    prelude::Backend,
    tensor::{
        module::{conv1d, conv_transpose1d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode},
        Tensor,
    },
};

#[derive(Debug, Config)]
pub struct AudioVaeConfig {
    #[config(default = 128)]
    encoder_dim: usize,
    #[config(default = "[2, 5, 8, 8]")]
    encoder_rates: [usize; 4],
    #[config(default = "Some(64)")]
    latent_dim: Option<usize>,
    #[config(default = 1536)]
    decoder_dim: usize,
    #[config(default = "[8, 8, 5, 2]")]
    decoder_rates: [usize; 4],
    #[config(default = true)]
    depthwise: bool,
    #[config(default = 16000)]
    sample_rate: usize,
    #[config(default = false)]
    use_noise_block: bool,
}

impl Default for AudioVaeConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 128,
            encoder_rates: [2, 5, 8, 8],
            latent_dim: Some(64),
            decoder_dim: 1536,
            decoder_rates: [8, 8, 5, 2],
            depthwise: true,
            sample_rate: 16000,
            use_noise_block: false,
        }
    }
}

impl AudioVaeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioVae<B> {
        let latent_dim = match self.latent_dim {
            Some(val) => val,
            None => self.encoder_dim * (2usize.pow(self.encoder_rates.len() as u32)),
        };
        AudioVae {
            encoder: CausalEncoderConfig::new()
                .with_d_model(self.encoder_dim)
                .with_latent_dim(latent_dim)
                .with_strides(self.encoder_rates)
                .with_depthwise(self.depthwise)
                .init(device),
            decoder: CausalDecoderConfig::new(latent_dim, self.decoder_dim, self.decoder_rates)
                .with_depthwise(self.depthwise)
                .with_use_noise_block(self.use_noise_block)
                .init(device),
            sample_rate: self.sample_rate,
            hop_length: self.encoder_rates.iter().product(),
            latent_dim,
            chunk_size: self.encoder_rates.iter().product(),
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioVae<B: Backend> {
    encoder: CausalEncoder<B>,
    decoder: CausalDecoder<B>,
    pub sample_rate: usize,
    hop_length: usize,
    pub latent_dim: usize,
    pub chunk_size: usize,
}

impl<B: Backend> AudioVae<B> {
    pub fn preprocess(&self, audio_date: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let sample_rate = match sample_rate {
            Some(val) => val,
            None => self.sample_rate,
        };

        let pad_to = self.hop_length;
        let length = audio_date.dims()[2];
        let right_pad = ((length as f32 / pad_to as f32).ceil()) as usize * pad_to - length;
        audio_date.pad((right_pad, 0, 0, 0), PadMode::Constant(0.0))
    }
    pub fn decode(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(z)
    }

    pub fn encode(&self, audio_data: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let audio_data = if audio_data.dims().len() == 2 {
            audio_data.unsqueeze_dim(1)
        } else {
            audio_data
        };
        let audio_data = self.preprocess(audio_data, sample_rate);
        self.encoder.forward(audio_data).mu
    }
}

#[derive(Debug, Config)]
pub struct CausalEncoderConfig {
    #[config(default = 64)]
    d_model: usize,
    #[config(default = 32)]
    latent_dim: usize,
    #[config(default = "[2, 4, 8, 8]")]
    strides: [usize; 4],
    #[config(default = false)]
    depthwise: bool,
}

impl CausalEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoder<B> {
        let mut block = vec![CausalEncoderLayerType::WNCausalConv1d(
            WNCausalConv1dConfig::new(1, self.d_model, 7)
                .with_padding(3)
                .init(device),
        )];
        let mut d_model_n = self.d_model;
        for stride in self.strides {
            d_model_n *= 2;
            let groups = if self.depthwise { d_model_n / 2 } else { 1 };
            block.push(CausalEncoderLayerType::CausalEncoderBlock(
                CausalEncoderBlockConfig::new()
                    .with_output_dim(d_model_n)
                    .with_stride(stride)
                    .with_groups(groups)
                    .init(device),
            ));
        }

        CausalEncoder {
            block,
            fc_mu: WNCausalConv1dConfig::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
            fc_logvar: WNCausalConv1dConfig::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
        }
    }
}

#[derive(Debug)]
pub struct EncoderOutput<B: Backend> {
    pub hidden_state: Tensor<B, 3>,
    pub mu: Tensor<B, 3>,
    pub logvar: Tensor<B, 3>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderLayerType<B: Backend> {
    WNCausalConv1d(WNCausalConv1d<B>),
    CausalEncoderBlock(CausalEncoderBlock<B>),
}

impl<B: Backend> CausalEncoderLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::CausalEncoderBlock(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoder<B: Backend> {
    fc_mu: WNCausalConv1d<B>,
    fc_logvar: WNCausalConv1d<B>,
    block: Vec<CausalEncoderLayerType<B>>,
}

impl<B: Backend> CausalEncoder<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> EncoderOutput<B> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }

        EncoderOutput {
            hidden_state: x.clone(),
            mu: self.fc_mu.forward(x.clone()),
            logvar: self.fc_logvar.forward(x),
        }
    }
}

#[derive(Debug, Config)]
pub struct CausalDecoderConfig {
    input_channel: usize,
    channels: usize,
    rates: [usize; 4],
    #[config(default = "false")]
    depthwise: bool,
    #[config(default = 1)]
    d_out: usize,
    #[config(default = "false")]
    use_noise_block: bool,
}

impl CausalDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoder<B> {
        let mut model = if self.depthwise {
            vec![
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(self.input_channel, self.input_channel, 7)
                        .with_padding(3)
                        .with_groups(self.input_channel)
                        .init(device),
                ),
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(self.input_channel, self.channels, 1).init(device),
                ),
            ]
        } else {
            vec![CausalDecoderLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.input_channel, self.channels, 7)
                    .with_padding(3)
                    .init(device),
            )]
        };

        let mut output_dim = 0;
        for (i, stride) in self.rates.iter().enumerate() {
            let input_dim = self.channels / 2usize.pow(i as u32);
            output_dim = self.channels / 2usize.pow(i as u32 + 1);
            let groups = if self.depthwise { output_dim } else { 1 };

            model.push(CausalDecoderLayerType::CausalDecoderBlock(
                CausalDecoderBlockConfig::new()
                    .with_input_dim(input_dim)
                    .with_output_dim(output_dim)
                    .with_stride(*stride)
                    .with_groups(groups)
                    .with_use_noise_block(self.use_noise_block)
                    .init(device),
            ));
        }

        model.push(CausalDecoderLayerType::Snake1d(
            Snake1dConfig::new(output_dim).init(device),
        ));
        model.push(CausalDecoderLayerType::WNCausalConv1d(
            WNCausalConv1dConfig::new(output_dim, self.d_out, 7)
                .with_padding(3)
                .init(device),
        ));
        model.push(CausalDecoderLayerType::Tanh(Tanh::new()));

        CausalDecoder { model }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum CausalDecoderLayerType<B: Backend> {
    WNCausalConv1d(WNCausalConv1d<B>),
    CausalDecoderBlock(CausalDecoderBlock<B>),
    Snake1d(Snake1d<B>),
    Tanh(Tanh),
}

impl<B: Backend> CausalDecoderLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::CausalDecoderBlock(val) => val.forward(x),
            Self::Tanh(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoder<B: Backend> {
    model: Vec<CausalDecoderLayerType<B>>,
}

impl<B: Backend> CausalDecoder<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in self.model.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct CausalDecoderBlockConfig {
    #[config(default = 16)]
    pub input_dim: usize,
    #[config(default = 8)]
    pub output_dim: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 1)]
    pub groups: usize,
    #[config(default = "false")]
    pub use_noise_block: bool,
}

impl CausalDecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoderBlock<B> {
        let mut block = vec![
            CausalDecoderBlockLayerType::Snake1d(Snake1dConfig::new(self.input_dim).init(device)),
            CausalDecoderBlockLayerType::WNCausalTransposeConv1d(
                WNCausalTransposeConv1dConfig::new(
                    self.input_dim,
                    self.output_dim,
                    2 * self.stride,
                    self.stride,
                    (self.stride as f32 / 2.0).ceil() as usize,
                    self.stride % 2,
                )
                .init(device),
            ),
        ];
        if self.use_noise_block {
            block.push(CausalDecoderBlockLayerType::NoiseBlock(
                NoiseBlockConfig::new(self.output_dim).init(device),
            ))
        }

        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(1)
                .with_groups(self.groups)
                .init(device),
        ));
        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(3)
                .with_groups(self.groups)
                .init(device),
        ));

        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(9)
                .with_groups(self.groups)
                .init(device),
        ));

        CausalDecoderBlock { block }
    }
}

#[derive(Module, Debug)]
pub enum CausalDecoderBlockLayerType<B: Backend> {
    Snake1d(Snake1d<B>),
    WNCausalTransposeConv1d(WNCausalTransposeConv1d<B>),
    NoiseBlock(NoiseBlock<B>),
    CausalResidualUnit(CausalResidualUnit<B>),
}

impl<B: Backend> CausalDecoderBlockLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalTransposeConv1d(val) => val.forward(x),
            Self::NoiseBlock(val) => val.forward(x),
            Self::CausalResidualUnit(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoderBlock<B: Backend> {
    pub block: Vec<CausalDecoderBlockLayerType<B>>,
}

impl<B: Backend> CausalDecoderBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct NoiseBlockConfig {
    dim: usize,
}

impl NoiseBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NoiseBlock<B> {
        NoiseBlock {
            linear: WNCausalConv1dConfig::new(self.dim, self.dim, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NoiseBlock<B: Backend> {
    linear: WNCausalConv1d<B>,
}

impl<B: Backend> NoiseBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [B, C, T] = x.dims();
        let noise = Tensor::random([B, 1, T], Default::default(), &x.device());
        let h = self.linear.forward(x.clone());
        let n = noise * h;
        x + n
    }
}

#[derive(Debug, Config)]
pub struct WNCausalTransposeConv1dConfig {
    input_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    #[config(default = true)]
    bias: bool,
}

impl WNCausalTransposeConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalTransposeConv1d<B> {
        let conv = ConvTranspose1dConfig::new([self.input_dim, self.output_dim], self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_padding_out(self.output_padding)
            .init(device);

        let v = conv.weight.clone();

        let g = burn::module::Param::from_tensor(
            v.val()
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .sum_dim(1)
                .sqrt(),
        );

        WNCausalTransposeConv1d {
            weight_v: v,
            weight_g: g,
            bias: if self.bias { conv.bias } else { None },
            stride: self.stride,
            kernel_size: self.kernel_size,
            padding: self.padding,
            output_padding: self.output_padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalTransposeConv1d<B: Backend> {
    pub weight_g: burn::module::Param<Tensor<B, 3>>,
    pub weight_v: burn::module::Param<Tensor<B, 3>>,
    pub bias: Option<burn::module::Param<Tensor<B, 1>>>,
    pub stride: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub output_padding: usize,
}

impl<B: Backend> WNCausalTransposeConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Materialize: see the note in WNCausalConv1d::forward.
        let device = x.device();
        let x = Tensor::from_data(x.to_data(), &device);
        let v = self.weight_v.val();

        let norm = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();

        let v_hat = v / (norm + 1e-12);

        let w = self.weight_g.val() * v_hat;

        let w = Tensor::from_data(w.to_data(), &device);

        let out = conv_transpose1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            ConvTransposeOptions::<1> {
                stride: [self.stride],
                padding: [0],
                padding_out: [0],
                dilation: [1],
                groups: 1,
            },
        );

        let trim = self.padding * 2 - self.output_padding;
        let length = out.dims()[2];
        // Materialize the trimmed view so downstream convs read a standard
        // layout.
        let trimmed = out.narrow(2, 0, length - trim);
        Tensor::from_data(trimmed.to_data(), &device)
    }
}

#[derive(Debug, Config)]
pub struct WNCausalConv1dConfig {
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 1)]
    groups: usize,
    #[config(default = 0)]
    padding: usize,
    #[config(default = true)]
    bias: bool,
}

impl WNCausalConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalConv1d<B> {
        let conv = Conv1dConfig::new(self.channels_in, self.channels_out, self.kernel_size)
            .with_groups(self.groups)
            .init(device);

        let v = conv.weight.clone();

        let g = burn::module::Param::from_tensor(
            v.val()
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .sum_dim(1)
                .sqrt(),
        );

        WNCausalConv1d {
            weight_v: v,
            weight_g: g,
            bias: if self.bias { conv.bias } else { None },
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            padding: self.padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalConv1d<B: Backend> {
    pub weight_g: burn::module::Param<Tensor<B, 3>>,
    pub weight_v: burn::module::Param<Tensor<B, 3>>,
    pub bias: Option<burn::module::Param<Tensor<B, 1>>>,
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl<B: Backend> WNCausalConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Materialize up front: callers may hand us narrow/slice views whose
        // non-standard memory layout the ndarray conv kernels mis-read (same
        // root cause as the weight materialization below).
        let x: Tensor<B, 3> = Tensor::from_data(x.to_data(), &x.device());
        let v = self.weight_v.val();

        let norm = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();

        let v_hat = v / (norm + 1e-12);

        let w = self.weight_g.val() * v_hat;

        let w = Tensor::from_data(w.to_data(), &x.device());

        let device = x.device();
        let padded = x.pad((self.padding * 2, 0, 0, 0), PadMode::Constant(0.0));
        // Materialize the padded buffer too: pad results carry a non-standard
        // layout that the ndarray conv kernel mis-reads (see note above).
        let x: Tensor<B, 3> = Tensor::from_data(padded.to_data(), &device);

        let opts = ConvOptions::<1> {
            stride: [self.stride],
            padding: [0],
            dilation: [self.dilation],
            groups: self.groups,
        };
        conv1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            opts,
        )
    }
}

#[derive(Debug, Config)]
pub struct CausalEncoderBlockConfig {
    #[config(default = 16)]
    output_dim: usize,
    input_dim: Option<usize>,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalEncoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoderBlock<B> {
        let input_dim = match self.input_dim {
            Some(val) => val,
            None => self.output_dim / 2,
        };

        let block = vec![
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(1)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(3)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(9)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::Snake1d(Snake1dConfig::new(input_dim).init(device)),
            CausalEncoderBlockLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(input_dim, self.output_dim, 2 * self.stride)
                    .with_padding((self.stride as f32 / 2.0).ceil() as usize)
                    .with_stride(self.stride)
                    .init(device),
            ),
        ];

        CausalEncoderBlock { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderBlockLayerType<B: Backend> {
    CausalResidualUnit(CausalResidualUnit<B>),
    Snake1d(Snake1d<B>),
    WNCausalConv1d(WNCausalConv1d<B>),
}

impl<B: Backend> CausalEncoderBlockLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::CausalResidualUnit(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoderBlock<B: Backend> {
    block: Vec<CausalEncoderBlockLayerType<B>>,
}

impl<B: Backend> CausalEncoderBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct CausalResidualUnitConfig {
    #[config(default = 16)]
    dim: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 7)]
    kernel: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalResidualUnitConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalResidualUnit<B> {
        let pad = ((7 - 1) * self.dilation) / 2;
        let block = vec![
            CausalResidualUnitLayerType::Snake1d(Snake1dConfig::new(self.dim).init(device)),
            CausalResidualUnitLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.dim, self.dim, self.kernel)
                    .with_dilation(self.dilation)
                    .with_padding(pad)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalResidualUnitLayerType::Snake1d(Snake1dConfig::new(self.dim).init(device)),
            CausalResidualUnitLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.dim, self.dim, 1).init(device),
            ),
        ];
        CausalResidualUnit { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum CausalResidualUnitLayerType<B: Backend> {
    Snake1d(Snake1d<B>),
    WNCausalConv1d(WNCausalConv1d<B>),
}

impl<B: Backend> CausalResidualUnitLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalResidualUnit<B: Backend> {
    pub block: Vec<CausalResidualUnitLayerType<B>>,
}

impl<B: Backend> CausalResidualUnit<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = x.clone();

        for layer in self.block.iter() {
            y = layer.forward(y);
        }

        let pad = (x.dims()[2] - y.dims()[2]) / 2;
        let r = pad..x.dims()[2] - pad;
        assert!(pad == 0);
        let x = x.slice_dim(2, r);
        x + y
    }
}

#[derive(Debug, Config)]
pub struct Snake1dConfig {
    channels: usize,
}

impl Snake1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Snake1d<B> {
        Snake1d {
            alpha: burn::module::Param::from_tensor(Tensor::ones(
                &vec![1, self.channels, 1],
                device,
            )),
        }
    }
}

#[derive(Module, Debug)]
pub struct Snake1d<B: Backend> {
    alpha: burn::module::Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val();
        x.clone() + (alpha.clone() + 1e-9).recip() * (alpha * x).sin().powi_scalar(2)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `audiovae_encode_decode_roundtrip_*` exercises the architecture end-to-end
// against uninitialized (zero) weights. The point isn't fidelity — untrained
// weights give garbage output — it's to catch the regression class where the
// forward graph changes shape, NaNs out, or panics on a real input tensor.
//
// The AudioVAE V2 port (the next VoxCPM2 blocker) will reuse this same harness:
// add a `audiovae_v2_encode_decode_roundtrip_*` pair and assert the new module
// also passes.

#[cfg(test)]
#[cfg(feature = "voxcpm")]
mod tests {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    use super::*;

    fn build_vae() -> (AudioVae<NdArray>, usize, usize) {
        let device = NdArrayDevice::Cpu;
        let cfg = AudioVaeConfig::default();
        let sample_rate = cfg.sample_rate;
        let hop_length = cfg.encoder_rates.iter().product::<usize>(); // 2*5*8*8 = 640
        let vae = cfg.init(&device);
        (vae, sample_rate, hop_length)
    }

    #[test]
    fn audiovae_encode_decode_shape_preserved_with_random_weights() {
        let (vae, sample_rate, hop_length) = build_vae();

        // 0.25 s at 16 kHz — small enough to be fast on CI, large enough to
        // exercise the encoder's full stride stack (640-sample hop).
        let n_samples = sample_rate / 4;
        let x: Tensor<NdArray, 3> = Tensor::zeros([1, 1, n_samples], &NdArrayDevice::Cpu);

        let z = vae.encode(x, Some(sample_rate));
        // Encoder output: [B, latent_dim, T/hop_length]
        assert_eq!(z.dims()[0], 1, "batch dim preserved");
        assert_eq!(z.dims()[1], vae.latent_dim, "latent dim matches config");
        let expected_latent_len = (n_samples + hop_length - 1) / hop_length; // ceil
        assert_eq!(
            z.dims()[2], expected_latent_len,
            "latent time dim should be ceil(samples/hop_length)"
        );

        let y = vae.decode(z);
        // Decoder output should match the padded input length (decode mirrors encode).
        assert_eq!(y.dims()[0], 1, "batch dim preserved");
        assert_eq!(y.dims()[1], 1, "single audio channel preserved");
        let out_len = y.dims()[2];
        let padded_in = expected_latent_len * hop_length;
        assert_eq!(
            out_len, padded_in,
            "decode output should mirror the padded input length"
        );
    }

    #[test]
    fn audiovae_encode_on_short_input_does_not_panic() {
        let (vae, sample_rate, hop_length) = build_vae();
        // 100 samples — far smaller than one hop block (640). The encoder
        // should still produce a valid latent (1 frame after right-padding).
        let n_samples = 100;
        let x: Tensor<NdArray, 3> = Tensor::zeros([1, 1, n_samples], &NdArrayDevice::Cpu);
        let z = vae.encode(x, Some(sample_rate));
        assert_eq!(z.dims()[0], 1);
        assert_eq!(z.dims()[1], vae.latent_dim);
        assert_eq!(z.dims()[2], 1, "single hop frame after right-padding");
        let _ = hop_length; // silence unused warning if test ever gets compiled out
    }
}

// TEMPORARY PROBE - not for commit.
// 1) Does the INCREMENTAL (streaming) decoder keep WeVisOne-VL's coordinate
//    tokens? 2) How many tokens is the official raw response?
fn main() {
    let path = std::env::args().nth(1).expect("tokenizer.json path");
    let tok = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ids: Vec<u32> = vec![151669, 151670, 151682, 152670, 152672, 151645, 151668];
    println!("decode(skip=true)  -> {:?}", tok.decode(&ids, true));

    let mut ds = tok.decode_stream(true);
    let mut streamed = String::new();
    for id in &ids {
        match ds.step(*id) {
            Ok(Some(s)) => {
                println!("  step {id} -> {s:?}");
                streamed.push_str(&s);
            }
            Ok(None) => println!("  step {id} -> None"),
            Err(e) => println!("  step {id} -> error {e}"),
        }
    }
    println!("decode_stream      -> {:?}", streamed);

    let official = "<ref>dog</ref><box><13><274><399><949>,<379><222><620><954>,<581><489><996><997>,<637><61><834><539></box>, <ref>chair</ref><box><416><0><586><208>,<708><0><832><139></box>, <ref>bush</ref><box><0><0><327><259></box>";
    let enc = tok.encode(official, false).expect("encode official");
    println!("official raw response token count = {}", enc.get_ids().len());
    println!("official ids = {:?}", enc.get_ids());
}

import { defineConfig } from 'vitepress'

// Bilingual site: English is the root locale, 简体中文 lives under /zh/.
// Each locale carries its own themeConfig (nav / sidebar / UI labels);
// the default theme auto-localizes its built-in UI strings from the
// per-locale `lang` field.
export default defineConfig({
  // GitHub Pages site URL: https://ibloc.github.io/neuron/. Without this,
  // every emitted asset resolves to /assets/... and the deployed site 404s.
  // Override locally with `vitepress --base=/` (set in the build script
  // below) for `vitepress dev` / `vitepress preview`.
  base: '/neuron/',
  title: 'neuron',
  description: 'Apache-2.0 Rust inference runtime for LLM / TTS / ASR / Rerank / Embedding / Vision — OpenAI- and Anthropic-compatible',
  cleanUrls: true,
  lastUpdated: true,
  appearance: 'dark',
  // Always ignore dead links at build time — link rot is now caught by the
  // `link-check` job in .github/workflows/docs.yml. Keeping `true` here
  // avoids spurious CI failures from transient HTTPS errors during build.
  ignoreDeadLinks: true,

  // Exclude internal-only design specs from the public site. The
  // docs/superpowers/ directory is git-ignored (per project policy) but
  // VitePress still walks docs/ for markdown to compile, so we have to
  // tell it to skip superpowers/ explicitly. See root /AGENTS.md and
  // shadow/AGENTS.md for the design-spec version-control rationale.
  srcExclude: ['**/superpowers/**'],

  head: [
    ['meta', { name: 'theme-color', content: '#1f6feb' }],
    ['meta', { property: 'og:title', content: 'neuron — Neural Inference Runtime' }],
    ['meta', { property: 'og:description', content: 'Apache-2.0 Rust inference runtime. LLM / TTS / ASR / Rerank / Embedding / Vision. OpenAI + Anthropic compatible. Library-only, embed into any Rust host.' }],
    ['meta', { property: 'og:type', content: 'website' }],
  ],

  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      themeConfig: {
        siteTitle: 'neuron',

        nav: [
          { text: 'Guide', link: '/guide/intro' },
          { text: 'Concepts', link: '/concepts/architecture' },
          { text: 'How-to', link: '/howto/embed-into-host' },
          { text: 'API', link: '/reference/api' },
          {
            text: 'v0.1.0',
            items: [
              { text: 'License (Apache-2.0)', link: '/license' },
            ],
          },
        ],

        sidebar: {
          '/guide/': [
            {
              text: 'Introduction',
              items: [
                { text: 'What is neuron?', link: '/guide/intro' },
                { text: 'Quick Start', link: '/guide/quick-start' },
              ],
            },
          ],
          '/concepts/': [
            {
              text: 'Concepts',
              items: [
                { text: 'Architecture', link: '/concepts/architecture' },
                { text: 'Capabilities', link: '/concepts/capabilities' },
                { text: 'Runtime Traits', link: '/concepts/runtime-traits' },
                { text: 'Embedding in Hosts', link: '/concepts/embedding' },
                { text: 'Engine Selection', link: '/concepts/engines' },
                { text: 'Feature Flags', link: '/concepts/feature-flags' },
                { text: 'The Model Scheduler', link: '/concepts/scheduler' },
              ],
            },
          ],
          '/howto/': [
            {
              text: 'How-to Guides',
              items: [
                { text: 'Embed neuron into a Rust Host', link: '/howto/embed-into-host' },
                { text: 'Multi-source Model Registry', link: '/howto/multi-source-registry' },
                { text: 'Vision Tasks (detect/classify/segment/pose)', link: '/howto/vision-tasks' },
                { text: 'LoRA Adapter Registration', link: '/howto/lora-adapters' },
                { text: 'OCR (text detection + recognition)', link: '/howto/ocr' },
              ],
            },
          ],
          '/reference/': [
            {
              text: 'API Reference',
              items: [
                { text: 'OpenAI-compatible Endpoints', link: '/reference/api' },
                { text: 'Anthropic Messages v1', link: '/reference/anthropic' },
                { text: 'Environment Variables', link: '/reference/env-vars' },
                { text: 'ServerState Fields', link: '/reference/server-state' },
              ],
            },
          ],
        },

        socialLinks: [
          { icon: 'github', link: 'https://github.com/ibloc/neuron' },
        ],

        footer: {
          message: 'Released under the Apache License 2.0',
          copyright: `Copyright © 2026 Neuron Contributors`,
        },

        editLink: {
          pattern: 'https://github.com/ibloc/neuron/edit/main/docs/:path',
          text: 'Edit this page on GitHub',
        },

        search: {
          provider: 'local',
        },

        outline: {
          level: [2, 3],
          label: 'On this page',
        },

        docFooter: {
          prev: 'Previous',
          next: 'Next',
        },
      },
    },

    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      description: 'Apache-2.0 Rust 推理运行时——LLM / TTS / ASR / Rerank / Embedding / Vision / OCR,OpenAI 与 Anthropic 线格式兼容',
      themeConfig: {
        siteTitle: 'neuron',

        nav: [
          { text: '指南', link: '/zh/guide/intro' },
          { text: '概念', link: '/zh/concepts/architecture' },
          { text: '教程', link: '/zh/howto/embed-into-host' },
          { text: 'API', link: '/zh/reference/api' },
          {
            text: 'v0.1.0',
            items: [
              { text: '许可证 (Apache-2.0)', link: '/license' },
            ],
          },
        ],

        sidebar: {
          '/zh/guide/': [
            {
              text: '介绍',
              items: [
                { text: 'neuron 是什么?', link: '/zh/guide/intro' },
                { text: '快速开始', link: '/zh/guide/quick-start' },
              ],
            },
          ],
          '/zh/concepts/': [
            {
              text: '核心概念',
              items: [
                { text: '架构', link: '/zh/concepts/architecture' },
                { text: '能力面(Capabilities)', link: '/zh/concepts/capabilities' },
                { text: 'Runtime Trait', link: '/zh/concepts/runtime-traits' },
                { text: '嵌入宿主进程', link: '/zh/concepts/embedding' },
                { text: '引擎选择', link: '/zh/concepts/engines' },
                { text: 'Feature Flags', link: '/zh/concepts/feature-flags' },
                { text: '模型调度器', link: '/zh/concepts/scheduler' },
              ],
            },
          ],
          '/zh/howto/': [
            {
              text: '实操教程',
              items: [
                { text: '嵌入 Rust 宿主程序', link: '/zh/howto/embed-into-host' },
                { text: '多源模型仓库', link: '/zh/howto/multi-source-registry' },
                { text: '视觉任务(detect/classify/segment/pose)', link: '/zh/howto/vision-tasks' },
                { text: 'LoRA 适配器注册', link: '/zh/howto/lora-adapters' },
                { text: 'OCR(文字检测 + 识别)', link: '/zh/howto/ocr' },
              ],
            },
          ],
          '/zh/reference/': [
            {
              text: 'API 参考',
              items: [
                { text: 'OpenAI 兼容端点', link: '/zh/reference/api' },
                { text: 'Anthropic Messages v1', link: '/zh/reference/anthropic' },
                { text: '环境变量', link: '/zh/reference/env-vars' },
                { text: 'ServerState 字段', link: '/zh/reference/server-state' },
              ],
            },
          ],
        },

        socialLinks: [
          { icon: 'github', link: 'https://github.com/ibloc/neuron' },
        ],

        footer: {
          message: '基于 Apache License 2.0 发布',
          copyright: `Copyright © 2026 Neuron Contributors`,
        },

        editLink: {
          pattern: 'https://github.com/ibloc/neuron/edit/main/docs/:path',
          text: '在 GitHub 上编辑此页',
        },

        search: {
          provider: 'local',
          options: {
            translations: {
              button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
              modal: {
                noResultsText: '未找到相关结果',
                resetButtonTitle: '清除查询条件',
                displayDetails: '显示详细列表',
                footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
              },
            },
          },
        },

        outline: {
          level: [2, 3],
          label: '本页目录',
        },

        docFooter: {
          prev: '上一篇',
          next: '下一篇',
        },
      },
    },
  },

  // No markdown.container overrides: the default theme localizes the
  // tip/warning/danger/info labels from each locale's `lang` field.
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
    lineNumbers: false,
  },
})

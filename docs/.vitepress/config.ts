import { defineConfig } from 'vitepress'

export default defineConfig({
  // GitHub Pages site URL: https://ibloc.github.io/neuron/. Without this,
  // every emitted asset resolves to /assets/... and the deployed site 404s.
  // Override locally with `vitepress --base=/` (set in the build script
  // below) for `vitepress dev` / `vitepress preview`.
  base: '/neuron/',
  title: 'neuron',
  description: 'Apache-2.0 Rust inference runtime for LLM / TTS / ASR / Rerank / Embedding / Vision — OpenAI- and Anthropic-compatible',
  lang: 'en-US',
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

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
    lineNumbers: false,
    container: {
      tipLabel: 'Tip',
      warningLabel: 'Warning',
      dangerLabel: 'Danger',
      infoLabel: 'Info',
      detailsLabel: 'Details',
    },
  },

  editLink: {
    pattern: 'https://github.com/ibloc/neuron/edit/main/docs/:path',
    text: 'Edit this page on GitHub',
  },
})

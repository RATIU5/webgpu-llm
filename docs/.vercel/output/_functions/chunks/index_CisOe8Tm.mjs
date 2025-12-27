import { Y as createVNode, I as Fragment, b2 as __astro_tag_component__ } from './astro/server_DUeoxbwr.mjs';
import { d as $$CardGrid, e as $$Card } from './Code_BHUdyGqO.mjs';

const frontmatter = {
  "title": "RATIU5' TypeGPU Docs",
  "description": "Learn about TypeGPU and WebGPU from these docs and AI",
  "template": "splash",
  "hero": {
    "tagline": "Learn about TypeGPU and WebGPU from these docs and AI",
    "actions": [{
      "text": "Get Started",
      "link": "/getting-started/",
      "icon": "right-arrow"
    }]
  }
};
function getHeadings() {
  return [{
    "depth": 2,
    "slug": "next-steps",
    "text": "Next steps"
  }];
}
function _createMdxContent(props) {
  const {Fragment: Fragment$1} = props.components || ({});
  if (!Fragment$1) _missingMdxReference("Fragment");
  return createVNode(Fragment, {
    children: [createVNode(Fragment$1, {
      "set:html": "<div class=\"sl-heading-wrapper level-h2\"><h2 id=\"next-steps\">Next steps</h2><a class=\"sl-anchor-link\" href=\"#next-steps\"><span aria-hidden=\"true\" class=\"sl-anchor-icon\"><svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\"><path fill=\"currentcolor\" d=\"m12.11 15.39-3.88 3.88a2.52 2.52 0 0 1-3.5 0 2.47 2.47 0 0 1 0-3.5l3.88-3.88a1 1 0 0 0-1.42-1.42l-3.88 3.89a4.48 4.48 0 0 0 6.33 6.33l3.89-3.88a1 1 0 1 0-1.42-1.42Zm8.58-12.08a4.49 4.49 0 0 0-6.33 0l-3.89 3.88a1 1 0 0 0 1.42 1.42l3.88-3.88a2.52 2.52 0 0 1 3.5 0 2.47 2.47 0 0 1 0 3.5l-3.88 3.88a1 1 0 1 0 1.42 1.42l3.88-3.89a4.49 4.49 0 0 0 0-6.33ZM8.83 15.17a1 1 0 0 0 1.1.22 1 1 0 0 0 .32-.22l4.92-4.92a1 1 0 0 0-1.42-1.42l-4.92 4.92a1 1 0 0 0 0 1.42Z\"></path></svg></span><span class=\"sr-only\">Section titled “Next steps”</span></a></div>\n"
    }), createVNode($$CardGrid, {
      children: [createVNode($$Card, {
        title: "Read the docs",
        icon: "document",
        "set:html": "<p>Begin reading the <a href=\"/getting-started/\">documentation</a> and learning about\nTypeGPU and WebGPU</p>"
      }), createVNode($$Card, {
        title: "Install the CLI",
        icon: "setting",
        "set:html": "<p>Using AI to learn? <a href=\"/installation/\">Install the CLI</a> to make the learning\nprocess easier.</p>"
      })]
    })]
  });
}
function MDXContent(props = {}) {
  const {wrapper: MDXLayout} = props.components || ({});
  return MDXLayout ? createVNode(MDXLayout, {
    ...props,
    children: createVNode(_createMdxContent, {
      ...props
    })
  }) : _createMdxContent(props);
}
function _missingMdxReference(id, component) {
  throw new Error("Expected " + ("component" ) + " `" + id + "` to be defined: you likely forgot to import, pass, or provide it.");
}

const url = "src/content/docs/index.mdx";
const file = "/Users/john.memmott/Developer/webgpu-llm/docs/src/content/docs/index.mdx";
const Content = (props = {}) => MDXContent({
  ...props,
  components: { Fragment: Fragment, ...props.components, },
});
Content[Symbol.for('mdx-component')] = true;
Content[Symbol.for('astro.needsHeadRendering')] = !Boolean(frontmatter.layout);
Content.moduleId = "/Users/john.memmott/Developer/webgpu-llm/docs/src/content/docs/index.mdx";
__astro_tag_component__(Content, 'astro:jsx');

export { Content, Content as default, file, frontmatter, getHeadings, url };

# Data design

## Contract status

This is an evolving conceptual contract for generated image pairs. JSON examples are illustrative; storage may later use JSONL, Parquet, or another versioned format. Incompatible changes require a schema-version change and documentation of their effect on existing experiments.

## Regression example

```json
{
  "schema_version": "1",
  "sample_id": "design2code_0042_font_size_001",
  "source_page_id": "0042",
  "split": "train",
  "reference_image": "processed/images/0042/reference.png",
  "candidate_image": "processed/images/0042/font_size_001.png",
  "is_regression": true,
  "mutation": {
    "mutation_family": "typography",
    "mutation_type": "font_size",
    "target": "#hero h2",
    "before_value": "16px",
    "after_value": "20px",
    "parameters": {
      "delta_px": 4
    }
  },
  "target_bbox": [412, 201, 318, 44],
  "render": {
    "browser": "chromium",
    "viewport_width": 1280,
    "viewport_height": 800,
    "device_scale_factor": 1.0
  }
}
```

## Benign example

```json
{
  "schema_version": "1",
  "sample_id": "design2code_0042_benign_001",
  "source_page_id": "0042",
  "split": "train",
  "reference_image": "processed/images/0042/reference.png",
  "candidate_image": "processed/images/0042/benign_001.png",
  "is_regression": false,
  "variation_type": "identical_rerender",
  "render": {
    "browser": "chromium",
    "viewport_width": 1280,
    "viewport_height": 800,
    "device_scale_factor": 1.0
  }
}
```

Benign cross-browser examples may eventually need separate `reference_render` and `candidate_render` records. That extension should be made when the team defines which cross-browser differences are valid benign training examples.

## Label policy

- A controlled CSS/DOM mutation that intentionally changes the visible design is labelled as a regression.
- Small magnitude does not make a mutation benign; for example, a one-pixel font-size change may still violate the intended specification.
- Benign/no-change samples are created through a separate path and carry a `variation_type`.
- Severity and human acceptability are not part of the initial binary label contract.

See [ADR 0002](adr/0002-regression-and-benign-label-policy.md).

## Split policy

The split unit is the source webpage:

```text
source webpage -> exactly one of train, validation, or internal test
```

Reference renders, every mutation, and every benign variant derived from that page remain in the same split. Split assignment should be generated once from a recorded seed, versioned as a manifest, and reused by all model comparisons. See [ADR 0003](adr/0003-source-page-disjoint-splits.md).

## Mutation taxonomy

The taxonomy is provisional. `Implemented` means available in the main branch with tests; therefore no mutation operator is currently implemented.

| Family | Mutation type | Status |
|---|---|---|
| Typography | `font_size`, `font_weight`, `line_height`, `letter_spacing` | Planned |
| Appearance | `color`, `opacity`, `border`, `border_radius` | Planned |
| Layout | `spacing`, `alignment`, `position`, `width`, `height` | Planned |
| Content/component | `missing_element`, `wrong_icon`, `wrong_asset` | Planned |
| Media | `aspect_ratio` | Deferred |

The next milestone should choose only one mutation type. Numerical distributions, selector rules, and the eventual number of accepted mutations per page remain unresolved and must not be duplicated as constants across scripts.

## Quality gates

A candidate pair should be accepted only after checking, at minimum:

- valid HTML/page load and candidate screenshot;
- matching expected image dimensions;
- required target location is available;
- nonzero visible pixel change for regression mutations;
- no dominant unrelated asset or page-load failure; and
- changed pixels are reasonably consistent with the intended target.

Passing these automated checks is not enough to scale generation. A representative pilot must also be reviewed manually.

## Path and identity rules

- `sample_id` is unique within a dataset version.
- `source_page_id` is stable across all derivatives of a page.
- Artifact paths stored in metadata are relative POSIX-style paths, even when generated on Windows.
- Absolute paths and parent traversal (`..`) are not portable and are rejected by the scaffold.
- Dataset manifests record their schema version and upstream dataset revision.

## Existing exploratory notebook

The `zehao-data-mutation-pipeline` branch contains an early Colab notebook that demonstrates HTML mutation and Playwright rendering ideas. It is not yet compatible with this contract: it uses Colab-specific absolute paths, runtime dependency installation, unseeded randomness, and in some cases assigns intentional CSS mutations a benign label. Preserve it as exploratory evidence, but bring useful logic across only after aligning it with the ADRs and adding provenance and quality checks.

# matgpr License Strategy

This note captures the licensing decision that must be resolved before adding a
final `LICENSE` file to the public `matgpr` repository.

## Desired Terms

Current preferred direction:

- free use for academic research,
- paid license for commercial use,
- required citation when users publish papers using `matgpr`,
- no redistribution,
- no modifications.

## Important Clarification

These terms are not open source under the Open Source Initiative definition.
The OSI definition requires free redistribution, allowance for derived works,
and no restriction against fields of endeavor such as business use.

The correct label for the current direction is closer to:

- source-available,
- academic/research-use license,
- dual commercial license,
- proprietary license with published source.

Use the term `open source` only if the final license allows commercial use,
redistribution, and modification under an OSI-compliant license.

## Practical Options

### Option A: True Open Source

Use a standard license such as BSD-3-Clause, MIT, or Apache-2.0.

Pros:

- easiest for academic adoption,
- easiest for package managers,
- easiest for contributors,
- compatible with JOSS-style publication expectations.

Cons:

- cannot prohibit commercial use,
- cannot prohibit redistribution,
- cannot prohibit modification.

Commercial revenue would come from hosted GenMatics platform access,
enterprise support, consulting, private features, or managed workflows.

### Option B: Source-Available Academic License

Use a custom license that allows academic research use but restricts commercial
use, redistribution, and modification.

Pros:

- better aligned with the stated commercial-control goal,
- preserves leverage for a paid enterprise license.

Cons:

- not open source,
- harder for contributors and institutions to approve,
- may block use in some universities, companies, or package ecosystems,
- should be drafted or reviewed by legal counsel.

### Option C: Dual License

Use a restricted academic/source-available license for public access and a
separate commercial license for companies.

Pros:

- aligns with free academic use plus paid commercial use,
- gives a clear path for industry users,
- can still keep code visible for trust-building.

Cons:

- requires license administration,
- requires clear boundaries for academic, nonprofit, startup, and industrial
  research uses,
- should be reviewed by legal counsel.

## Citation

Add `CITATION.cff` independent of the final license. Citation is best handled
through:

- `CITATION.cff`,
- README citation section,
- documentation citation page,
- Zenodo DOI after first release.

Forcing citation through a restrictive software license is possible in custom
terms, but it is better to provide clear scholarly citation metadata and
request citation in docs. Legal enforceability should be reviewed separately.

## Recommended Next Action

Do not add a final `LICENSE` file until the business model is decided.

Recommended path:

1. Add `CITATION.cff` now.
2. Add README citation guidance now.
3. Decide between true open source and source-available dual licensing.
4. If restricted terms are required, ask counsel to draft the license.
5. Add the final `LICENSE` before the first public release tag.

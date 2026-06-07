# Release Checklist

## Before GitHub release

- [ ] Replace `https://github.com/your-org/MICA` in `CITATION.cff`.
- [ ] Confirm license choice with all contributors.
- [ ] Decide whether the pretrained `mica_top47_bundle` should be committed or attached as a release asset.
- [ ] Run `npm pack --dry-run`.
- [ ] Run `npm install -g .` from a clean checkout.
- [ ] Run `mica doctor`.
- [ ] Run `mica inspect`.
- [ ] Run `mica validate --input examples/example_full98_features.csv`.
- [ ] Run `mica run --input examples/example_full98_features.csv --output /tmp/mica_predictions.csv`.
- [ ] Run `mica screen --input /tmp/mica_predictions.csv --output /tmp/mica_screen.csv`.

## Before npm publish

```bash
npm login
npm pack --dry-run
npm publish --access public
```


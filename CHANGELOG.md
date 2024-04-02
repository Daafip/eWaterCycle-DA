## Changelog

### v0.0.1
working basic version
#### v0.0.2
_(forgot to tag this properly so its under the tag v0.0.1)_
- upgrade to new version: added support for on the fly for Data assimilation
    - You can now call `.update()` without requiring DA support 
    - This means that any model can use the programmed data structure
    - calling `.update(assimilate=False)` will no longer run the DA code
- smaller changes, bug fixes and additions to docs
#### v0.0.3
- All user installed plugins are now availible by default.
- `.update(assimilate=False)` is now standard when calling `.update()` to match changes from v0.0.2
- Seperate list `KNOWN_WORKING_MODELS_DA` check whether to continue with assimilating.
- small change to `config_specific_actions`, but this is a WIP 

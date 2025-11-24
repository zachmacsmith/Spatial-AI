# Archived Files

This directory contains deprecated files that have been replaced by the new modular architecture.

## Old TestingClass Files

These 5 files have been replaced by the modular `video_processing/` system:

- **TestingClass.py** → Use `PRESET_BASIC`
- **TestingClass2.py** → Use `PRESET_OBJECTS`
- **TestingClass3.py** → Use `PRESET_RELATIONSHIPS`
- **TestingClassIntegrated.py** → Use `PRESET_HTML_ANALYSIS`
- **TestingClassFINAL.py** → Use `PRESET_FULL`

## Why Archived?

- **Redundancy**: 2,902 total lines with 75-80% duplication
- **Maintainability**: Bug fixes required changes in 5 places
- **Flexibility**: Hard to test different configurations

## New System Benefits

- **Zero duplication**: Modular architecture
- **40+ parameters**: Complete control
- **Batch tracking**: Link outputs to exact configs
- **Extensible**: Registry pattern for new methods
- **Multi-model**: Claude, Gemini, OpenAI + YOLO variants

## Migration Guide

See main README.md for usage examples.

## Preservation

These files are kept for reference and to ensure no functionality is lost during migration.

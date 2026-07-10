export function normalizeRemovedWidgetValues(values, layouts) {
  if (!Array.isArray(values)) {
    return values;
  }
  const removedIndexes = layouts[values.length];
  if (!removedIndexes) {
    return values;
  }
  const normalized = [...values];
  for (const index of [...removedIndexes].sort((left, right) => right - left)) {
    normalized.splice(index, 1);
  }
  return normalized;
}

const LEGACY_FLUX_EDIT_MODES = new Set(["text_to_image", "single_reference", "multi_reference"]);
const FLUX_LAYOUTS = {
  8: [2, 3, 6, 7],
  9: [4, 5],
  10: [2, 5, 6],
  13: [4, 5],
};
const Z_IMAGE_LAYOUTS = {
  5: [0, 2, 3],
  9: [0, 2, 3],
};

export function normalizeFluxSettingsWidgetValues(values) {
  if (Array.isArray(values) && values.length === 11) {
    return LEGACY_FLUX_EDIT_MODES.has(values[2])
      ? normalizeRemovedWidgetValues(values, { 11: [2, 3, 6, 7] })
      : values;
  }
  return normalizeRemovedWidgetValues(values, FLUX_LAYOUTS);
}

export function normalizeZImageSettingsWidgetValues(values) {
  return normalizeRemovedWidgetValues(values, Z_IMAGE_LAYOUTS);
}

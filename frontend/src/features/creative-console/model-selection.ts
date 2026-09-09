import type { ModelRouteDTO } from "../../entities/model/types";

export function uniqueModelsByPublicID<T extends { publicId: string }>(models: readonly T[]): T[] {
  const seen = new Set<string>();
  return models.filter((model) => {
    if (seen.has(model.publicId)) return false;
    seen.add(model.publicId);
    return true;
  });
}

export function filterVoiceModels(
  models: readonly ModelRouteDTO[],
  subMode: "tts" | "stt",
): ModelRouteDTO[] {
  const matched = models.filter((item) => (
    subMode === "tts"
      ? item.capability === "tts" || item.capability === "realtime"
      : item.capability === "stt"
  ));
  return uniqueModelsByPublicID(matched);
}

export function resolveActiveModel<T extends { publicId: string }>(
  candidates: readonly T[],
  currentModel: string,
): string {
  if (candidates.some((item) => item.publicId === currentModel)) {
    return currentModel;
  }
  return candidates[0]?.publicId ?? "";
}

export function shouldSynchronizeActiveModel(
  activeModel: string,
  currentModel: string,
): boolean {
  return Boolean(activeModel && activeModel !== currentModel);
}

export function updateSelectedModel<T extends Record<string, string>, K extends keyof T = keyof T>(
  current: T,
  mode: K,
  model: string,
): T {
  if (current[mode] === model) {
    return current;
  }
  return { ...current, [mode]: model };
}

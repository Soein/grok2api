import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ModelRouteDTO } from "../../entities/model/types";
import {
  filterVoiceModels,
  resolveActiveModel,
  shouldSynchronizeActiveModel,
  uniqueModelsByPublicID,
  updateSelectedModel,
} from "./model-selection.ts";

function createMockRoute(
  publicId: string,
  capability: ModelRouteDTO["capability"],
  id = `route-${publicId}`,
): ModelRouteDTO {
  return {
    id,
    publicId,
    capability,
    provider: "grok_console",
    upstreamModel: publicId,
    origin: "catalog",
    enabled: true,
    accountIds: ["acc-1"],
    bindingMode: false,
    supportedAccounts: 1,
    syncedAccounts: 1,
    totalAccounts: 1,
    capabilityKnown: true,
    available: true,
  };
}

describe("model-selection", () => {
  describe("uniqueModelsByPublicID", () => {
    it("deduplicates routes sharing the same publicId and preserves order", () => {
      const routes = [
        createMockRoute("grok-voice-1", "tts", "route-1"),
        createMockRoute("grok-voice-2", "tts", "route-2"),
        createMockRoute("grok-voice-1", "tts", "route-3"),
      ];
      const result = uniqueModelsByPublicID(routes);
      assert.equal(result.length, 2);
      assert.equal(result[0]?.id, "route-1");
      assert.equal(result[1]?.id, "route-2");
    });
  });

  describe("filterVoiceModels", () => {
    it("matches tts and realtime capabilities when subMode is tts", () => {
      const routes = [
        createMockRoute("voice-tts", "tts"),
        createMockRoute("voice-rt", "realtime"),
        createMockRoute("voice-stt", "stt"),
        createMockRoute("text-chat", "chat"),
      ];
      const ttsModels = filterVoiceModels(routes, "tts");
      assert.deepEqual(
        ttsModels.map((m) => m.publicId),
        ["voice-tts", "voice-rt"],
      );
    });

    it("matches only stt capability when subMode is stt", () => {
      const routes = [
        createMockRoute("voice-tts", "tts"),
        createMockRoute("voice-stt", "stt"),
      ];
      const sttModels = filterVoiceModels(routes, "stt");
      assert.deepEqual(
        sttModels.map((m) => m.publicId),
        ["voice-stt"],
      );
    });

    it("returns an empty list when no route matches the requested subMode", () => {
      const routes = [createMockRoute("voice-stt", "stt")];
      const ttsModels = filterVoiceModels(routes, "tts");
      assert.deepEqual(ttsModels, []);
    });
  });

  describe("resolveActiveModel", () => {
    it("keeps the current model if it exists in the candidate list", () => {
      const candidates = [{ publicId: "alpha" }, { publicId: "beta" }];
      assert.equal(resolveActiveModel(candidates, "beta"), "beta");
    });

    it("falls back to the first candidate if the current model is not found", () => {
      const candidates = [{ publicId: "alpha" }, { publicId: "beta" }];
      assert.equal(resolveActiveModel(candidates, "gamma"), "alpha");
      assert.equal(resolveActiveModel(candidates, ""), "alpha");
    });

    it("returns empty string if the candidate list is empty", () => {
      assert.equal(resolveActiveModel([], "alpha"), "");
      assert.equal(resolveActiveModel([], ""), "");
    });
  });

  describe("shouldSynchronizeActiveModel", () => {
    it("blocks synchronizing when activeModel is empty", () => {
      assert.equal(shouldSynchronizeActiveModel("", "grok-stt"), false);
      assert.equal(shouldSynchronizeActiveModel("", "grok-voice"), false);
      assert.equal(shouldSynchronizeActiveModel("", ""), false);
    });

    it("blocks synchronizing when activeModel already matches current model", () => {
      assert.equal(shouldSynchronizeActiveModel("grok-voice", "grok-voice"), false);
    });

    it("permits synchronizing only when activeModel is non-empty and differs from current model", () => {
      assert.equal(shouldSynchronizeActiveModel("grok-stt", "grok-voice"), true);
      assert.equal(shouldSynchronizeActiveModel("grok-voice", ""), true);
    });
  });

  describe("updateSelectedModel", () => {
    it("returns the exact same object reference if the model has not changed", () => {
      const current = { chat: "c1", image: "i1", video: "v1", voice: "voice-1" };
      const next = updateSelectedModel(current, "voice", "voice-1");
      assert.equal(next, current);
    });

    it("returns a new object with the updated mode when the model differs", () => {
      const current = { chat: "c1", image: "i1", video: "v1", voice: "voice-1" };
      const next = updateSelectedModel(current, "voice", "voice-2");
      assert.notEqual(next, current);
      assert.deepEqual(next, { chat: "c1", image: "i1", video: "v1", voice: "voice-2" });
    });
  });

  describe("integrated production selection flow", () => {
    it("prevents render loops in STT-only route by suppressing empty-model sync to parent", () => {
      const permittedModels = [createMockRoute("grok-stt", "stt")];
      const voiceChoices = uniqueModelsByPublicID(permittedModels);

      // Parent initializes with empty selection; fallback resolves to the STT route
      const initialSelectedVoice = "";
      const parentEffectiveVoice = resolveActiveModel(voiceChoices, initialSelectedVoice);
      assert.equal(parentEffectiveVoice, "grok-stt");

      // VoicePanel mounts with default subMode "tts"
      const filteredTTS = filterVoiceModels(permittedModels, "tts");
      assert.deepEqual(filteredTTS, []);
      const panelActiveModel = resolveActiveModel(filteredTTS, parentEffectiveVoice);
      assert.equal(panelActiveModel, "");

      // Guard blocks syncing empty active model back to parent
      const shouldSync = shouldSynchronizeActiveModel(panelActiveModel, parentEffectiveVoice);
      assert.equal(shouldSync, false);
    });

    it("prevents render loops in TTS-only route when switching to STT submode", () => {
      const permittedModels = [createMockRoute("grok-voice", "tts")];
      const voiceChoices = uniqueModelsByPublicID(permittedModels);

      const parentEffectiveVoice = resolveActiveModel(voiceChoices, "grok-voice");
      assert.equal(parentEffectiveVoice, "grok-voice");

      // User switches VoicePanel to STT submode where no matching route exists
      const filteredSTT = filterVoiceModels(permittedModels, "stt");
      assert.deepEqual(filteredSTT, []);
      const panelActiveModel = resolveActiveModel(filteredSTT, parentEffectiveVoice);
      assert.equal(panelActiveModel, "");

      // Guard blocks syncing empty active model back to parent
      const shouldSync = shouldSynchronizeActiveModel(panelActiveModel, parentEffectiveVoice);
      assert.equal(shouldSync, false);
    });

    it("synchronizes model cleanly in 1 step when switching between compatible routes", () => {
      const permittedModels = [
        createMockRoute("grok-voice", "tts"),
        createMockRoute("grok-stt", "stt"),
      ];
      const voiceChoices = uniqueModelsByPublicID(permittedModels);

      let parentState = { chat: "", image: "", video: "", voice: "grok-voice" };
      let parentEffectiveVoice = resolveActiveModel(voiceChoices, parentState.voice);
      assert.equal(parentEffectiveVoice, "grok-voice");

      // User switches VoicePanel to STT submode
      const filteredSTT = filterVoiceModels(permittedModels, "stt");
      const panelActiveSTT = resolveActiveModel(filteredSTT, parentEffectiveVoice);
      assert.equal(panelActiveSTT, "grok-stt");

      // Step 1: Guard detects difference and permits update
      assert.equal(shouldSynchronizeActiveModel(panelActiveSTT, parentEffectiveVoice), true);
      const nextState = updateSelectedModel(parentState, "voice", panelActiveSTT);
      assert.notEqual(nextState, parentState);
      assert.equal(nextState.voice, "grok-stt");
      parentState = nextState;

      // Step 2: Parent re-renders with new selection; guard now detects equality and blocks further updates
      parentEffectiveVoice = resolveActiveModel(voiceChoices, parentState.voice);
      assert.equal(parentEffectiveVoice, "grok-stt");
      const nextPanelActiveSTT = resolveActiveModel(filteredSTT, parentEffectiveVoice);
      assert.equal(nextPanelActiveSTT, "grok-stt");
      assert.equal(shouldSynchronizeActiveModel(nextPanelActiveSTT, parentEffectiveVoice), false);
    });

    it("preserves referential equality when an update would not change the selected model", () => {
      const state = { chat: "c1", image: "i1", video: "v1", voice: "grok-voice" };
      const next = updateSelectedModel(state, "voice", "grok-voice");
      assert.equal(next, state);
    });
  });
});

# AGENTS.md — V2. Repetición de contexto variado

## 1. Regla crítica de aislamiento (V1 vs V2)
- Este repositorio (`MiguelLeonardo/V2-Repeticion-de-contexto-variado`) es **completamente independiente** de V1 (`MiguelLeonardo/Languaje-learning`).
- **NUNCA** modifiques, hagas push ni sobrescribas archivos, `index.html` o ramas (`main`, `gh-pages`, `v1`) del repositorio V1 (`MiguelLeonardo/Languaje-learning`).
- Todos los cambios de UI (`backend/app/static/index.html`, `public/index.html` y `gh-pages` `index.html`) pertenecen exclusivamente a **este repositorio V2**.

## 2. Reglas técnicas heredadas de V1
- **Captura de voz móvil y Word Trainer**:
  - `SpeechRecognition.start()` debe ejecutarse de forma síncrona dentro del clic del usuario antes de `startWordAudioRecorder()`.
  - Esperar `MediaRecorder.onstop` (`stopWordAudioRecorderAndWait`) antes de re-renderizar la UI para no perder `state.wordAudioUrls[idx]`.
  - Mantener fallback de Android (`sf_android_no_parallel_rec`).
- **Calidad y pruebas**:
  - Cobertura mínima de `pytest`: **85%** (`make lint && make content-validate && cd backend && .venv/bin/python -m pytest tests/`).
  - Cero errores de `ruff`.
  - Verificar sintaxis JS de los bloques `<script>` antes de hacer commit.
- **Currículo**:
  - 384 escenarios YAML en `content/scenarios/` (8 módulos × 8 escenarios por nivel CEFR `A1`–`C2`, 5 frases por escenario).

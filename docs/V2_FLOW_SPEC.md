# V2. Repetición de contexto variado — Especificación del flujo

## Diferencia clave frente a V1
- **V1 ("Repetición espaciada")**: Las fases 1, 2 y 3 practicaban la **misma oración** y solo la fase 4 variaba el contexto.
- **V2 ("Repetición de contexto variado")**: Los modismos/expresiones objetivo de cada ejercicio se recorren a través de **4 situaciones distintas** en las primeras 4 fases, culminando en la conversación abierta:
  1. **1. Sombra** (`shadowing`) — **Contexto 1**: Oración base del escenario, audio automático y resaltado de modismos.
  2. **2. Practica** (`guided`) — **Contexto 2**: Nueva situación diferente usando los mismos modismos, con ~33% letras guía únicamente en los modismos y audio guía.
  3. **3. Completa** (`completion`) — **Contexto 3**: Tercera situación diferente usando los mismos modismos en espacios `____` y audio guía.
  4. **4. Aplica** (`recall`) — **Contexto 4**: Cuarta situación diferente integrando los modismos practicados.
  5. **5. Conversa** (`roleplay`) — **Conversación abierta** integrando los modismos de los 4 contextos.

## Siguiente paso recomendado al iniciar V2
1. Separar `FALLBACK_SCENARIOS` de `index.html` a un `scenarios.json` externo cargado por `fetch` (manteniendo `index.html` ligero y fácil de editar).
2. Extender el esquema de frases (`backend/app/content/schema.py` y `content/scenarios/*.yaml`) para soportar 4 contextos explícitos por frase, actualizando `backend/app/services/exercises.py` y `buildLocalLesson`.

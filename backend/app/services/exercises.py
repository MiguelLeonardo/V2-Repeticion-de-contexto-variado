"""Builds exercises with decreasing visual support from curriculum content.

Stages (Speak-like progression, all answered by voice):
  shadowing   -> listen & repeat with full text + audio (main input exercise)
  recall      -> FULL support: text, translation, glossary, visual hint
  guided      -> PARTIAL support: translation + first-letter hints, visual hint
  completion  -> LOW support: sentence with blanks, no translation
  roleplay    -> SCRIPT support: open AI chat with suggested lines (see roleplay.py)
"""

from enum import StrEnum

from pydantic import BaseModel

from app.content.repository import ContentRepository
from app.content.schema import GlossaryEntry, Phrase, Scenario


class ExerciseType(StrEnum):
    SHADOWING = "shadowing"
    RECALL = "recall"
    GUIDED = "guided"
    COMPLETION = "completion"


class SupportLevel(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    LOW = "low"


STAGE_CONFIG: dict[ExerciseType, tuple[SupportLevel, float, str]] = {
    ExerciseType.SHADOWING: (
        SupportLevel.FULL,
        0.70,
        "Escucha el audio (se reproduce solo) y repite la frase imitando el ritmo. Fíjate en el modismo resaltado.",
    ),
    ExerciseType.GUIDED: (
        SupportLevel.PARTIAL,
        0.75,
        "Practica la frase en un contexto diferente ayudándote con las letras incompletas únicamente en el modismo.",
    ),
    ExerciseType.COMPLETION: (
        SupportLevel.LOW,
        0.80,
        "Completa los espacios ____ usando el mismo modismo en una tercera situación (puedes usar la guía de audio).",
    ),
    ExerciseType.RECALL: (
        SupportLevel.FULL,
        0.85,
        "Aplica las expresiones aprendidas en un cuarto contexto diferente guiándote por las letras incompletas.",
    ),
}
LESSON_ORDER = [ExerciseType.SHADOWING, ExerciseType.GUIDED, ExerciseType.COMPLETION, ExerciseType.RECALL]


class ExercisePrompt(BaseModel):
    instruction_es: str
    display_text: str | None = None
    translation: str | None = None
    glossary: list[GlossaryEntry] = []
    visual_hint: str | None = None
    audio_text: str | None = None  # text the client sends to TTS for model audio
    focus_words: list[str] = []


class Exercise(BaseModel):
    id: str
    type: ExerciseType
    support: SupportLevel
    scenario_id: str
    phrase_id: str
    prompt: ExercisePrompt
    expected_text: str
    pass_threshold: float
    response_mode: str = "speech"  # every exercise is answered with the microphone


def exercise_id(scenario_id: str, phrase_id: str, kind: ExerciseType) -> str:
    return f"{scenario_id}.{phrase_id}.{kind.value}"


def _partial_hint(text: str) -> str:
    """Reveal ~33% leading letters on each word (at least 2 chars for >=4-letter words)."""
    out = []
    for w in text.split():
        core = w.rstrip(".,!?")
        tail = w[len(core):]
        if len(core) <= 1:
            out.append(w)
        else:
            min_chars = 2 if len(core) >= 4 else 1
            count = min(len(core) - 1, max(min_chars, round(len(core) * 0.33)))
            out.append(core[:count] + "_" * (len(core) - count) + tail)
    return " ".join(out)


def _short_meaning(entry: GlossaryEntry | None, fallback: str) -> str:
    if not entry or not entry.meaning:
        return fallback
    m = entry.meaning.split("/")[0].split("(")[0].strip()
    if ":" in m:
        m = m.split(":", 1)[1].strip()
    return fallback if len(m) > 32 or not m else m


def _extract_targets(phrase: Phrase) -> tuple[str, str, str, str, list[str]]:
    g0 = phrase.glossary[0] if phrase.glossary else None
    g1 = phrase.glossary[1] if len(phrase.glossary) > 1 else None
    raw0 = (phrase.apply_answer or (g0.word if g0 else "") or "").strip()
    if not raw0:
        words = [w.strip(".,!?;:\"'()") for w in phrase.text.split()]
        raw0 = next((w for w in words if len(w) > 3), words[0] if words else "expression")
    raw1 = g1.word.strip() if g1 and g1.word and g1.word.strip().lower() != raw0.lower() else ""
    mean0 = _short_meaning(g0, raw0)
    mean1 = _short_meaning(g1, raw1) if raw1 else ""
    terms: list[str] = []
    for t in (raw0, raw1):
        if t:
            for part in t.split():
                clean = part.strip(".,!?;:\"'()")
                if clean:
                    terms.append(clean)
    return raw0, raw1, mean0, mean1, list(dict.fromkeys(terms))


def _idiom_token_mask(tokens: list[str], focus_terms: list[str]) -> list[bool]:
    mask = [False] * len(tokens)
    clean_tokens = [t.strip(".,!?;:\"'()").lower() for t in tokens]
    for term in focus_terms:
        tw = term.lower().strip(".,!?;:\"'()")
        if not tw:
            continue
        for i, ct in enumerate(clean_tokens):
            if ct == tw:
                mask[i] = True
    return mask


def _mask_sentence(sentence: str, focus_terms: list[str], mode: str) -> str:
    tokens = sentence.split()
    mask = _idiom_token_mask(tokens, focus_terms)
    if not any(mask) and tokens:
        mask[len(tokens) // 2] = True
    out = []
    for tok, is_target in zip(tokens, mask, strict=True):
        if not is_target:
            out.append(tok)
        elif mode == "cloze":
            core = tok.rstrip(".,!?")
            tail = tok[len(core):]
            out.append("____" + tail)
        else:
            out.append(_partial_hint(tok))
    return " ".join(out)


def _extract_vp(sentence: str, target: str) -> str:
    idx = sentence.lower().find(target.lower())
    if idx == -1:
        return target
    vp = sentence[idx:].strip()
    for sep in ("?", "!", ".", ";", ", please", " please"):
        if sep in vp:
            vp = vp.split(sep, 1)[0].strip()
    return vp or target


_TOPIC_CONTEXTS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "food-and-drink": {
        "places": [
            ("at the neighborhood café", "en la cafetería del barrio"),
            ("during lunch at the bistro", "durante el almuerzo en el bistró"),
            ("when ordering dinner", "al pedir la cena"),
            ("with the server at our table", "con el mesero en nuestra mesa"),
            ("at the bakery counter", "en el mostrador de la panadería"),
        ],
        "tasks": [
            ("ordering the tasting menu", "pedir el menú degustación"),
            ("splitting the dinner bill", "dividir la cuenta de la cena"),
            ("choosing a table on the patio", "elegir una mesa en la terraza"),
            ("asking about daily specials", "preguntar por los especiales del día"),
            ("customizing the coffee order", "personalizar el pedido de café"),
        ],
    },
    "travel-and-transport": {
        "places": [
            ("at the departure gate", "en la puerta de embarque"),
            ("at the central train station", "en la estación central de trenes"),
            ("before boarding the flight", "antes de abordar el vuelo"),
            ("at the ticket kiosk", "en el quiosco de boletos"),
            ("near the baggage claim area", "cerca de la zona de equipaje"),
        ],
        "tasks": [
            ("checking the platform number", "revisar el número de andén"),
            ("catching the express train", "tomar el tren expreso"),
            ("finding the airport shuttle", "encontrar el traslado al aeropuerto"),
            ("confirming the seat upgrade", "confirmar el cambio de asiento"),
            ("updating the travel itinerary", "actualizar el itinerario de viaje"),
        ],
    },
    "hotel-and-shopping": {
        "places": [
            ("at the hotel front desk", "en la recepción del hotel"),
            ("at the checkout counter", "en la caja de la tienda"),
            ("near the fitting rooms", "cerca de los probadores"),
            ("when calling room service", "al llamar al servicio a la habitación"),
            ("at customer service", "en atención al cliente"),
        ],
        "tasks": [
            ("completing the late check-in", "completar el registro tardío"),
            ("exchanging a jacket size", "cambiar la talla de una chaqueta"),
            ("requesting extra towels", "pedir toallas adicionales"),
            ("reviewing the store receipt", "revisar el recibo de la tienda"),
            ("booking a quieter room", "reservar una habitación más tranquila"),
        ],
    },
    "work-and-career": {
        "places": [
            ("during the weekly stand-up", "durante la reunión semanal"),
            ("on the client review call", "en la llamada de revisión con el cliente"),
            ("before the quarterly presentation", "antes de la presentación trimestral"),
            ("while updating the project board", "al actualizar el tablero del proyecto"),
            ("in the team retrospective", "en la retrospectiva del equipo"),
        ],
        "tasks": [
            ("aligning the sprint priorities", "alinear las prioridades del sprint"),
            ("preparing the status deck", "preparar la presentación de avance"),
            ("handing off the client brief", "entregar el resumen del cliente"),
            ("reviewing the budget forecast", "revisar la proyección de presupuesto"),
            ("onboarding the new analyst", "integrar al nuevo analista"),
        ],
    },
    "health-and-emergencies": {
        "places": [
            ("at the clinic reception", "en la recepción de la clínica"),
            ("when talking to the pharmacist", "al hablar con el farmacéutico"),
            ("during the medical check-up", "durante el chequeo médico"),
            ("at the urgent care desk", "en el mostrador de urgencias"),
            ("on the follow-up call with the nurse", "en la llamada de seguimiento con la enfermera"),
        ],
        "tasks": [
            ("explaining the symptoms clearly", "explicar los síntomas con claridad"),
            ("picking up the prescription", "recoger la receta médica"),
            ("scheduling a follow-up visit", "programar una cita de seguimiento"),
            ("checking the dosage instructions", "revisar las instrucciones de dosis"),
            ("updating the allergy form", "actualizar el formulario de alergias"),
        ],
    },
    "social-and-smalltalk": {
        "places": [
            ("over coffee with a friend", "tomando un café con un amigo"),
            ("at the weekend barbecue", "en la parrillada del fin de semana"),
            ("at the neighborhood gathering", "en la reunión del vecindario"),
            ("when catching up after work", "al ponerse al día después del trabajo"),
            ("at the dinner party", "en la cena con amigos"),
        ],
        "tasks": [
            ("planning the weekend hike", "planear la caminata del fin de semana"),
            ("sharing travel stories", "compartir historias de viaje"),
            ("recommending a local restaurant", "recomendar un restaurante local"),
            ("organizing the birthday dinner", "organizar la cena de cumpleaños"),
            ("introducing new neighbors", "presentar a nuevos vecinos"),
        ],
    },
    "grammar-and-connectors": {
        "places": [
            ("when comparing both options", "al comparar ambas opciones"),
            ("during the class discussion", "durante el debate en clase"),
            ("while explaining your reasoning", "al explicar tu razonamiento"),
            ("when summarizing the main point", "al resumir el punto principal"),
            ("while connecting two viewpoints", "al conectar dos puntos de vista"),
        ],
        "tasks": [
            ("structuring a clear argument", "estructurar un argumento claro"),
            ("contrasting two proposals", "contrastar dos propuestas"),
            ("clarifying the timeline", "aclarar el cronograma"),
            ("supporting the final decision", "respaldar la decisión final"),
            ("rephrasing the key takeaway", "reformular la idea clave"),
        ],
    },
    "daily-tech-and-media": {
        "places": [
            ("when setting up the Wi-Fi", "al configurar el Wi-Fi"),
            ("before joining the video call", "antes de entrar a la videollamada"),
            ("while troubleshooting the app", "al resolver un fallo en la aplicación"),
            ("when updating device settings", "al actualizar los ajustes del dispositivo"),
            ("while managing cloud backups", "al gestionar las copias en la nube"),
        ],
        "tasks": [
            ("securing the home router", "asegurar el router de casa"),
            ("syncing the shared calendar", "sincronizar el calendario compartido"),
            ("checking the privacy permissions", "revisar los permisos de privacidad"),
            ("clearing the browser cache", "limpiar la caché del navegador"),
            ("testing the microphone audio", "probar el audio del micrófono"),
        ],
    },
    "cv-dhl-gsc-tl": {
        "places": [
            ("in the global operations review", "en la revisión de operaciones globales"),
            ("during the DHL leadership panel", "durante el panel de liderazgo de DHL"),
            ("when presenting KPI improvements", "al presentar mejoras de KPIs"),
            ("in the UAT handover session", "en la sesión de entrega UAT"),
            ("during the automation briefing", "durante la reunión de automatización"),
        ],
        "tasks": [
            ("standardizing regional reporting", "estandarizar los reportes regionales"),
            ("stabilizing the EDI data flow", "estabilizar el flujo de datos EDI"),
            ("aligning the cross-functional team", "alinear al equipo multifuncional"),
            ("automating the recurring dashboard", "automatizar el tablero recurrente"),
            ("closing the audit action items", "cerrar los puntos de acción de auditoría"),
        ],
    },
}

_BARE_VERB_PREFIXES = (
    "have ", "call ", "make ", "take ", "get ", "feel ", "fit ", "break ", "hit ",
    "cut ", "keep ", "let ", "iron ", "move ", "sort ", "fill ", "pick ", "wake ",
    "hold ", "check ", "set ", "wrap ", "lock ", "bridge ", "scale ", "drive ",
    "step ", "stay ", "own ", "roll ", "flag ", "pull ", "push ", "turn ", "bring ",
    "run ", "map ", "streamline ", "fast-track ", "stay ", "work ", "build ",
)


def _lc_first(s: str) -> str:
    if len(s) > 1 and not s.startswith(("I ", "I'")):
        return s[0].lower() + s[1:]
    return s


def _cap_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _varied_context(
    phrase: Phrase,
    stage: ExerciseType,
    variant_idx: int,
    topic: str = "work-and-career",
) -> tuple[str, str]:
    """Build distinct, topic-aware, grammatically sound sentences for Contexts 2, 3, and 4."""
    t0, _, m0, _, _ = _extract_targets(phrase)
    lower0 = t0.lower()
    base_en = phrase.text.strip()
    base_es = phrase.translation.split("(")[0].strip().rstrip(".")
    ctx = _TOPIC_CONTEXTS.get(topic, _TOPIC_CONTEXTS["work-and-career"])
    places = ctx["places"]
    tasks = ctx["tasks"]

    stage_offset = 0 if stage == ExerciseType.GUIDED else (1 if stage == ExerciseType.COMPLETION else 2)
    place_en, place_es = places[(variant_idx + stage_offset) % len(places)]
    task_en, task_es = tasks[(variant_idx + stage_offset) % len(tasks)]

    # 1. Phrasal verbs
    if phrase.kind == "phrasal_verb":
        vp_text = _extract_vp(base_en, t0)
        vp_apply = t0
        if phrase.apply_sentence and "____" in phrase.apply_sentence:
            after = phrase.apply_sentence.split("____", 1)[1].strip().rstrip(".!?")
            if after and not after.startswith(","):
                vp_apply = f"{t0} {after}".split(",")[0].strip()
        if lower0.endswith("ing"):
            return (
                f"{_cap_first(place_en)}, we were focused on {vp_apply}.",
                f'{_cap_first(place_es)}, nos enfocamos en "{m0}".',
            )
        if stage == ExerciseType.GUIDED:
            return (
                f"{_cap_first(place_en)}, you often need to {vp_text}.",
                f'{_cap_first(place_es)}, a menudo necesitas "{m0}" ({base_es.lower()}).',
            )
        if stage == ExerciseType.COMPLETION:
            return (
                f"{_cap_first(place_en)}, make sure to {vp_apply}.",
                f'{_cap_first(place_es)}, asegúrate de "{m0}".',
            )
        return (
            f"{_cap_first(place_en)}, our team tries to {vp_apply} right away.",
            f'{_cap_first(place_es)}, nuestro equipo procura "{m0}" de inmediato.',
        )

    # 2. Idioms and expressions: only slot t0 directly when its grammatical role is verified
    if phrase.kind in ("idiom", "expression"):
        is_nominal = lower0.startswith(("a ", "an ", "the ", "piece of"))
        is_bare_verb = lower0.startswith(_BARE_VERB_PREFIXES)
        if is_nominal:
            if stage == ExerciseType.GUIDED:
                return (
                    f"{_cap_first(place_en)}, {_lc_first(task_en)} was {t0} for everyone.",
                    f'{_cap_first(place_es)}, {task_es} fue "{m0}" para todos.',
                )
            if stage == ExerciseType.COMPLETION:
                return (
                    f"After preparing well {place_en}, {task_en} felt like {t0}.",
                    f'Después de prepararnos bien {place_es}, {task_es} se sintió como "{m0}".',
                )
            return (
                f"With clear steps {place_en}, {task_en} became {t0}.",
                f'Con pasos claros {place_es}, {task_es} se volvió "{m0}".',
            )
        if is_bare_verb:
            if stage == ExerciseType.GUIDED:
                return (
                    f"{_cap_first(place_en)}, we decided to {t0} before continuing.",
                    f'{_cap_first(place_es)}, decidimos "{m0}" antes de continuar.',
                )
            if stage == ExerciseType.COMPLETION:
                return (
                    f"When {_lc_first(task_en)} {place_en}, it helps to {t0}.",
                    f'Al {task_es} {place_es}, conviene "{m0}".',
                )
            return (
                f"{_cap_first(place_en)}, everyone agreed to {t0} together.",
                f'{_cap_first(place_es)}, todos acordaron "{m0}" juntos.',
            )

        # For adverbial/prepositional/clausal idioms, use apply_sentence if available
        # or wrap the complete grammatical clause.
        apply_full = ""
        if phrase.apply_sentence and phrase.apply_answer:
            apply_full = phrase.apply_sentence.replace("____", phrase.apply_answer).strip()
        if stage == ExerciseType.COMPLETION and apply_full:
            return (
                f"{_cap_first(place_en)}, {_lc_first(apply_full)}",
                f"{_cap_first(place_es)}: {(phrase.apply_translation or base_es)}",
            )
        return (
            f"{_cap_first(place_en)}, {_lc_first(base_en)}",
            f"{_cap_first(place_es)}: {base_es}.",
        )

    # 3. Conversational phrases (kind == 'phrase'): concise, topic-matched spoken context without narrator boilerplate
    clause = _lc_first(base_en)
    if stage == ExerciseType.GUIDED:
        return (
            f"{_cap_first(place_en)}, {clause}",
            f"{_cap_first(place_es)}: «{base_es}».",
        )
    if stage == ExerciseType.COMPLETION:
        return (
            f"While {task_en}, {clause}",
            f"Al {task_es}: «{base_es}».",
        )
    return (
        f"Actually, {place_en}, {clause}",
        f"De hecho, {place_es}: «{base_es}».",
    )


def build_exercise(
    scenario: Scenario,
    phrase: Phrase,
    kind: ExerciseType,
    focus_words: list[str] | None = None,
    variant_idx: int = 0,
) -> Exercise:
    support, threshold, instruction = STAGE_CONFIG[kind]
    _, _, _, _, default_terms = _extract_targets(phrase)
    terms = focus_words or default_terms

    if kind == ExerciseType.SHADOWING:
        prompt = ExercisePrompt(
            instruction_es=instruction,
            display_text=phrase.text,
            translation=phrase.translation,
            glossary=phrase.glossary,
            visual_hint=phrase.visual_hint,
            audio_text=phrase.text,
            focus_words=terms,
        )
        expected = phrase.text
    elif kind == ExerciseType.GUIDED:
        if phrase.practice_sentence:
            ctx2_en = phrase.practice_sentence
            ctx2_es = phrase.practice_translation or phrase.translation
        else:
            ctx2_en, ctx2_es = _varied_context(phrase, ExerciseType.GUIDED, variant_idx, scenario.topic)
        prompt = ExercisePrompt(
            instruction_es=instruction,
            display_text=_mask_sentence(ctx2_en, terms, mode="partial"),
            translation=ctx2_es,
            glossary=phrase.glossary,
            visual_hint=phrase.visual_hint,
            audio_text=ctx2_en,
            focus_words=terms,
        )
        expected = ctx2_en
    elif kind == ExerciseType.COMPLETION:
        if phrase.complete_sentence:
            ctx3_en = phrase.complete_sentence
        else:
            ctx3_en, _ = _varied_context(phrase, ExerciseType.COMPLETION, variant_idx, scenario.topic)
        prompt = ExercisePrompt(
            instruction_es=instruction,
            display_text=_mask_sentence(ctx3_en, terms, mode="cloze"),
            translation=None,
            glossary=[],
            visual_hint=phrase.visual_hint,
            audio_text=ctx3_en,
            focus_words=terms,
        )
        expected = ctx3_en
    else:  # ExerciseType.RECALL (4. Aplica - Contexto 4)
        if phrase.apply_sentence:
            completed = phrase.apply_sentence.replace("____", phrase.apply_answer or "")
            masked = phrase.apply_sentence.replace("____", _partial_hint(phrase.apply_answer or ""))
            prompt = ExercisePrompt(
                instruction_es=instruction,
                display_text=masked,
                translation=phrase.apply_translation,
                glossary=phrase.glossary,
                visual_hint=phrase.visual_hint,
                audio_text=completed,
                focus_words=[phrase.apply_answer] if phrase.apply_answer else terms,
            )
            expected = completed
        else:
            ctx4_en, ctx4_es = _varied_context(phrase, ExerciseType.RECALL, variant_idx, scenario.topic)
            prompt = ExercisePrompt(
                instruction_es=instruction,
                display_text=_mask_sentence(ctx4_en, terms, mode="partial"),
                translation=ctx4_es,
                glossary=phrase.glossary,
                visual_hint=phrase.visual_hint,
                audio_text=ctx4_en,
                focus_words=terms,
            )
            expected = ctx4_en

    if focus_words:
        prompt.focus_words = focus_words
    return Exercise(
        id=exercise_id(scenario.id, phrase.id, kind),
        type=kind,
        support=support,
        scenario_id=scenario.id,
        phrase_id=phrase.id,
        prompt=prompt,
        expected_text=expected,
        pass_threshold=threshold,
    )


def resolve_exercise(repo: ContentRepository, ex_id: str) -> Exercise | None:
    parts = ex_id.split(".")
    if len(parts) != 3:
        return None
    scenario_id, phrase_id, kind = parts
    try:
        ex_type = ExerciseType(kind)
    except ValueError:
        return None
    scenario = repo.get(scenario_id)
    phrase = repo.phrase(scenario_id, phrase_id)
    if not scenario or not phrase:
        return None
    idx = next((i for i, p in enumerate(scenario.phrases) if p.id == phrase_id), 0)
    return build_exercise(scenario, phrase, ex_type, variant_idx=idx)


class LessonStage(BaseModel):
    type: ExerciseType
    support: SupportLevel
    exercises: list[Exercise]


class RoleplayScriptOverview(BaseModel):
    ai_role: str
    user_role: str
    setting: str
    target_phrases: list[str]
    user_missions: list[str]


class LessonPlan(BaseModel):
    scenario_id: str
    title: str
    cefr: str
    flow_mode: str = "varied_context_by_phrase"
    stages: list[LessonStage]
    roleplay_available: bool = True
    roleplay_script: RoleplayScriptOverview | None = None


def build_lesson(scenario: Scenario) -> LessonPlan:
    stages = [
        LessonStage(
            type=k,
            support=STAGE_CONFIG[k][0],
            exercises=[build_exercise(scenario, p, k, variant_idx=idx) for idx, p in enumerate(scenario.phrases)],
        )
        for k in LESSON_ORDER
    ]
    script = RoleplayScriptOverview(
        ai_role=scenario.roleplay.ai_role,
        user_role=scenario.roleplay.user_role,
        setting=scenario.roleplay.setting,
        target_phrases=[p.text for p in scenario.phrases],
        user_missions=[t.goal for t in scenario.roleplay.turns if t.speaker == "user" and t.goal],
    )
    return LessonPlan(
        scenario_id=scenario.id,
        title=scenario.title,
        cefr=scenario.cefr.value,
        stages=stages,
        roleplay_script=script,
    )


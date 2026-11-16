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


def _varied_context(phrase: Phrase, stage: ExerciseType, variant_idx: int) -> tuple[str, str]:
    """Build distinct natural sentences for Context 2 (guided), Context 3 (completion), and Context 4 (recall)."""
    t0, t1, m0, m1, _ = _extract_targets(phrase)
    lower0 = t0.lower()
    base_en = phrase.text.strip()
    base_es = phrase.translation.split("(")[0].strip().rstrip(".")

    # 1. Phrasal verbs (e.g. wake up, fill out, hold on, call back, sort out, pick up)
    if phrase.kind == "phrasal_verb":
        vp_text = _extract_vp(base_en, t0)
        vp_apply = t0
        if phrase.apply_sentence and "____" in phrase.apply_sentence:
            after = phrase.apply_sentence.split("____", 1)[1].strip().rstrip(".!?")
            if after and not after.startswith(","):
                vp_apply = f"{t0} {after}".split(",")[0].strip()
        if lower0.endswith("ing"):
            if stage == ExerciseType.GUIDED:
                return (
                    f"At the central station, my coworker was {vp_apply}.",
                    f'En la estación central, mi compañero estaba aplicando "{m0}".',
                )
            if stage == ExerciseType.COMPLETION:
                return (
                    f"During the afternoon break, we were {vp_text}.",
                    f'Durante la pausa de la tarde, estábamos practicando "{m0}".',
                )
            return (
                f"Right before the flight, several passengers were {vp_apply}.",
                f'Justo antes del vuelo, varios pasajeros estaban usando "{m0}".',
            )
        if stage == ExerciseType.GUIDED:
            opts = [
                (
                    f"During a busy morning at work, you often need to {vp_text}.",
                    f'En una mañana ocupada en el trabajo, a menudo necesitas "{m0}" ({base_es.lower()}).',
                ),
                (
                    f"When traveling with friends, we always try to {vp_text}.",
                    f'Al viajar con amigos, siempre intentamos "{m0}" en esta situación.',
                ),
                (
                    f"At the service counter, the staff asked us to {vp_text}.",
                    f'En el mostrador de atención, el personal nos pidió "{m0}".',
                ),
            ]
            return opts[variant_idx % len(opts)]
        if stage == ExerciseType.COMPLETION:
            opts = [
                (
                    f"Before finishing the shift, make sure to {vp_apply}.",
                    f'Antes de terminar el turno, asegúrate de "{m0}" según lo indicado.',
                ),
                (
                    f"Even on a hectic Monday, our team managed to {vp_apply}.",
                    f'Incluso en un lunes ajetreado, nuestro equipo logró "{m0}" a tiempo.',
                ),
                (
                    f"At the hotel reception, guests usually need to {vp_apply}.",
                    f'En la recepción del hotel, los huéspedes suelen necesitar "{m0}".',
                ),
            ]
            return opts[variant_idx % len(opts)]
        opts = [
            (
                f"Whenever a new client arrives, we {vp_apply} right away.",
                f'Siempre que llega un cliente nuevo, decidimos "{m0}" de inmediato.',
            ),
            (
                f"Before the meeting starts, everyone should {vp_apply}.",
                f'Antes de que empiece la reunión, todos deberían "{m0}".',
            ),
        ]
        return opts[variant_idx % len(opts)]

    # 2. Idioms and expressions (e.g. a piece of cake, on time, right away, fit like a glove, call it a day)
    if phrase.kind in ("idiom", "expression"):
        is_nominal = lower0.startswith(("a ", "an ", "the ", "piece of"))
        is_verb_idiom = lower0.startswith((
            "have ", "call ", "make ", "take ", "get ", "feel ", "fit ", "break ", "hit ", "cut ", "keep ", "let ",
        ))
        if stage == ExerciseType.GUIDED:
            if is_nominal:
                return (
                    f"Thanks to the clear guide, setting up the account was {t0} for the team.",
                    f'Gracias a la guía clara, configurar la cuenta fue "{m0}" para el equipo.',
                )
            if is_verb_idiom:
                return (
                    f"After finishing the main presentation, the team decided to {t0} together.",
                    f'Después de terminar la presentación principal, el equipo decidió "{m0}" juntos.',
                )
            return (
                f"At the downtown office, our colleagues handled the request {t0} this morning.",
                f'En la oficina del centro, nuestros colegas gestionaron la solicitud "{m0}" esta mañana.',
            )
        if stage == ExerciseType.COMPLETION:
            if is_nominal:
                return (
                    f"Once we practiced the steps, the final interview felt like {t0} today.",
                    f'Una vez que practicamos los pasos, la entrevista final se sintió como "{m0}" hoy.',
                )
            if is_verb_idiom:
                return (
                    f"Before the weekend trip starts, we always {t0} so everything goes well.",
                    f'Antes del viaje de fin de semana, siempre procuramos "{m0}" para que todo salga bien.',
                )
            return (
                f"During the afternoon check-in, the receptionist confirmed our booking {t0}.",
                f'Durante el registro de la tarde, la recepcionista confirmó nuestra reserva "{m0}".',
            )
        return (
            f"In the new project, everyone was glad to see the plan work {t0}.",
            f'En el nuevo proyecto, todos se alegraron de ver funcionar el plan "{m0}".',
        )

    # 3. Conversational phrases (kind == 'phrase'):
    # Vary the real-world situational context around the natural clause so target expressions stay 100% grammatical.
    clause = (
        base_en[0].lower() + base_en[1:]
        if len(base_en) > 1 and not base_en.startswith(("I ", "I'"))
        else base_en
    )
    if stage == ExerciseType.GUIDED:
        frames = [
            (
                f"At the hotel lounge, you turn to the staff and say: {clause}",
                f"En el salón del hotel, te diriges al personal y dices: «{base_es}».",
            ),
            (
                f"During your morning break downtown, you politely say: {clause}",
                f"Durante tu descanso de la mañana en el centro, dices con cortesía: «{base_es}».",
            ),
            (
                f"At the airport counter before boarding, you tell the agent: {clause}",
                f"En el mostrador del aeropuerto antes de abordar, dices al agente: «{base_es}».",
            ),
        ]
        return frames[variant_idx % len(frames)]
    if stage == ExerciseType.COMPLETION:
        frames = [
            (
                f"While visiting a new city with friends, you step up and say: {clause}",
                f"Mientras visitas una ciudad nueva con amigos, te acercas y dices: «{base_es}».",
            ),
            (
                f"At the central station desk, you smile and say: {clause}",
                f"En el mostrador de la estación central, sonríes y dices: «{base_es}».",
            ),
            (
                f"Before the afternoon meeting begins, you ask the host: {clause}",
                f"Antes de que empiece la reunión de la tarde, dices al anfitrión: «{base_es}».",
            ),
        ]
        return frames[variant_idx % len(frames)]
    frames = [
        (
            f"Later that evening at the reception desk, you naturally say: {clause}",
            f"Más tarde esa noche en recepción, dices con naturalidad: «{base_es}».",
        ),
        (
            f"When arriving at the main office, you clearly say: {clause}",
            f"Al llegar a la oficina principal, dices claramente: «{base_es}».",
        ),
        (
            f"During the business trip abroad, you confidently say: {clause}",
            f"Durante el viaje de negocios en el extranjero, dices con confianza: «{base_es}».",
        ),
    ]
    return frames[variant_idx % len(frames)]


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
            ctx2_en, ctx2_es = _varied_context(phrase, ExerciseType.GUIDED, variant_idx)
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
            ctx3_en, _ = _varied_context(phrase, ExerciseType.COMPLETION, variant_idx)
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
            ctx4_en, ctx4_es = _varied_context(phrase, ExerciseType.RECALL, variant_idx)
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


import json
import uuid
import os
from datetime import datetime, timezone
from datapizza.clients.openai import OpenAIClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchTextAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_COLLECTION_NAME = "Hackaton"
ENABLE_TEXT_BM25 = True

# Initialize the client with your API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

client = OpenAIClient(
    api_key=api_key,
    model="gpt-4o-mini",  # Default model
    system_prompt="You are a helpful assistant that analyzes students and returns structured summaries.",  # Optional
    temperature=0.3,  # Optional, controls randomness (0-2)
)

qdrant_client = QdrantClient(
    url="https://d33cbe98-febe-4bc0-8bf3-92fa0e8ec660.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Je9OZQpzgAHQl7uV06Fv9flTuFAeXT1tNq0tlqm9ETw",
)


def ensure_qdrant_collection(vector_size: int) -> None:
    try:
        qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # Ensure payload indexes exist for filtered fields
    numeric_fields = {
        "age",
        "financial_support_per_year",
        "financial_support_duration",
    }

    for field_name in (
        "dimension",
        "user_id",
        "field_of_study",
        "country",
        "languages",
        "age",
        "current_status",
        "job_role",
        "financial_support_per_year",
        "financial_support_duration",
        "financial_support_return",
    ):
        field_schema = "integer" if field_name in numeric_fields else "keyword"
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # Index may already exist or index creation may not be supported for this field; ignore
            pass

    for text_field in ("summary_field", "summary_behaviour"):
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=text_field,
                field_schema="text",
            )
        except Exception:
            pass


# Basic text response
def summarize_student(student_description: str) -> dict:
    prompt = f"""
You will receive a free-text self-description from a student. It may be short, informal, and in any language.
Your task is to extract only what is explicitly stated or is an extremely direct implication (e.g. “primo anno” → “first year of their degree”). Do not invent goals, preferences, or background details that are not clearly supported by the text.
Based ONLY on that description, write the following fields:

summary_field
2–4 sentences in English describing:
    what they study (degree, field, university/school name if mentioned),
    any work roles or positions they currently have (company names must be kept exactly as in the text),
    the kinds of domains, tools, or topics they say they like or work with,
    if they explicitly mention future goals or desired roles, include them; if not mentioned, do not guess.

summary_behaviour
2–4 sentences in English describing:
    work ethic, discipline, motivation, and reliability only if they are mentioned or clearly implied,
    how they balance study/work if they explicitly mention it,
    how they describe their style (e.g. “very dedicated”, “chaotic but creative”, “team player”) using the same nuance as the original text,
    do not add generic traits that are not clearly in the description.

Very important constraints:
Preserve all concrete facts and proper nouns.
    You MUST explicitly mention every institution, university, company, role, and specific interest the student names in at least one of the summaries when relevant.
No hallucinations.
    Do not infer long-term career plans, personality traits, or preferences that are not clearly supported.
    If the text does not state something (e.g. collaboration style, reliability), you may simply omit it rather than guessing.
No change of meaning.
    Stay faithful to the tone: if they sound casual or humorous, you can reflect that lightly, but keep the summaries clear and professional.
Do not exaggerate or downplay what they say.

Output format.
    Reply with a valid JSON object with exactly these two keys: "summary_field" and "summary_behaviour".
    The value of each key must be a single string.
    The response must be ONLY the JSON. Do NOT use markdown, backticks, or ```json fences. Do NOT add any explanation.

Student description:
""" + student_description + """
"""
    response = client.invoke(prompt)
    raw_text = response.text.strip()

    # Strip markdown code fences if the model still uses them
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    data = None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to recover JSON from within surrounding text
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        return {
            "summary_field": data.get("summary_field", ""),
            "summary_behaviour": data.get("summary_behaviour", ""),
        }

def store_student_field_in_qdrant(
    student_description: str,
    user_id: str,
    field_of_study: str | None = None,
    country: str | None = None,
    languages: list[str] | None = None,
    age: int | None = None,
    current_status: str | None = None,
    job_role: str | None = None,
    financial_support_per_year: int | None = None,
    financial_support_duration: int | None = None,
    financial_support_return: str | None = None,
) -> dict:
    summaries = summarize_student(student_description)

    field_text = summaries.get("summary_field") or student_description
    behaviour_text = summaries.get("summary_behaviour") or student_description

    embeddings = client.embed(
        [field_text, behaviour_text], model_name=EMBEDDING_MODEL
    )
    if not embeddings or len(embeddings) != 2:
        raise RuntimeError("Failed to generate embeddings for student summaries.")

    field_embedding, behaviour_embedding = embeddings

    if languages is None:
        languages = []

    created_at = datetime.now(timezone.utc).isoformat()

    vector_size = len(field_embedding)
    ensure_qdrant_collection(vector_size)

    field_point_id = str(uuid.uuid4())
    behaviour_point_id = str(uuid.uuid4())

    field_payload = {
        "dimension": "field",
        "user_id": user_id,
        "summary_field": summaries.get("summary_field"),
        "field_of_study": field_of_study,
        "age": age,
        "country": country,
        "languages": languages,
        "current_status": current_status,
        "job_role": job_role,
        "financial_support_per_year": financial_support_per_year,
        "financial_support_duration": financial_support_duration,
        "financial_support_return": financial_support_return,
        "created_at": created_at,
    }

    behaviour_payload = {
        "dimension": "behaviour",
        "user_id": user_id,
        "summary_behaviour": summaries.get("summary_behaviour"),
        "field_of_study": field_of_study,
        "age": age,
        "country": country,
        "languages": languages,
        "current_status": current_status,
        "job_role": job_role,
        "financial_support_per_year": financial_support_per_year,
        "financial_support_duration": financial_support_duration,
        "financial_support_return": financial_support_return,
        "created_at": created_at,
    }

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True,
        points=[
            PointStruct(
                id=field_point_id,
                vector=field_embedding,
                payload=field_payload,
            ),
            PointStruct(
                id=behaviour_point_id,
                vector=behaviour_embedding,
                payload=behaviour_payload,
            ),
        ],
    )

    return {
        "student_id": user_id,
        "ids": {
            "field": field_point_id,
            "behaviour": behaviour_point_id,
        },
        "summaries": summaries,
    }


def build_believer_queries(believer_text: str) -> dict:
    prompt = f"""
You will receive text written by a believer (mentor, manager, or company) describing what kind of students they prefer. It may be short, informal, and in any language.
Your task is to extract only what is explicitly stated or is an extremely direct implication. Do not invent preferences, behaviours, or constraints that are not clearly supported by the text.
Based ONLY on that text, write the following fields:

query_field
2–4 sentences in English describing:
    the preferred students’ fields of study (including university/school names if mentioned),
    any required or preferred skills, domains, or technical areas (e.g. AI, software development, data, finance),
    any preferred backgrounds or credentials (e.g. specific degree, specific program),
    if they explicitly mention types of roles or projects (e.g. startup projects, research, consulting), include them; if not, do not guess.

query_behaviour
1–4 sentences in English describing:
    behaviour, work ethic, reliability, collaboration style, discipline, or motivation only if they are mentioned or clearly implied,
    if the believer does not say anything about behaviour or work style, explicitly state that they do not specify behavioural preferences, instead of inventing them.

Very important constraints:
Preserve all concrete facts and proper nouns.
    You MUST explicitly mention every institution, university, company, role, and specific domain the believer names (e.g. “Bocconi”, “AI”, “software development”) in at least one of the fields when relevant.
No hallucinations.
    Do not infer personality traits, work styles, or extra requirements that are not clearly supported.
    Do not assume they want “team players”, “proactive students”, “disciplined” or “reliable” students unless the text clearly suggests this.
No change of meaning.
    Stay faithful to the believer’s tone and level of strictness (e.g. if they say “molto interessato a…”, reflect that strong interest, but do not turn it into a long list of invented requirements).
Do not generalize from one specific preference to a broad set of expectations.

Output format.
    Reply with a valid JSON object with exactly these two keys: "query_field" and "query_behaviour".
    The value of each key must be a single string.
    The response must be ONLY the JSON. Do NOT use markdown, backticks, or ```json fences. Do NOT add any explanation.

Believer text:
""" + believer_text + """
"""

    response = client.invoke(prompt)
    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    data = None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        return {
            "query_field": data.get("query_field", ""),
            "query_behaviour": data.get("query_behaviour", ""),
        }

    return {
        "query_field": raw_text,
        "query_behaviour": "",
    }


def match_students_for_believer(
    believer_text: str,
    top_k: int = 200,
    queries: dict | None = None,
    field_of_study: list[str] | None = None,
    country: list[str] | None = None,
    filter_languages: list[str] | None = None,
    current_status: list[str] | None = None,
    job_role: list[str] | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
    financial_support_per_year: int | None = None,
    financial_support_duration: int | None = None,
    financial_support_return: list[str] | None = None,
) -> dict:
    if queries is None:
        queries = build_believer_queries(believer_text)

    field_query_text = queries.get("query_field") or believer_text
    behaviour_query_text = queries.get("query_behaviour") or believer_text

    query_embeddings = client.embed(
        [field_query_text, behaviour_query_text], model_name=EMBEDDING_MODEL
    )
    if not query_embeddings or len(query_embeddings) != 2:
        raise RuntimeError("Failed to generate embeddings for believer queries.")

    field_query_vec, behaviour_query_vec = query_embeddings

    # Ensure collection and payload indexes exist before querying
    ensure_qdrant_collection(len(field_query_vec))

    field_must = [
        FieldCondition(
            key="dimension",
            match=MatchValue(value="field"),
        ),
    ]

    behaviour_must = [
        FieldCondition(
            key="dimension",
            match=MatchValue(value="behaviour"),
        ),
    ]

    if ENABLE_TEXT_BM25:
        field_must.append(
            FieldCondition(
                key="summary_field",
                match=MatchTextAny(text_any=field_query_text),
            )
        )
        behaviour_must.append(
            FieldCondition(
                key="summary_behaviour",
                match=MatchTextAny(text_any=behaviour_query_text),
            )
        )

    if field_of_study:
        condition = FieldCondition(
            key="field_of_study",
            match=MatchAny(any=field_of_study),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if country:
        condition = FieldCondition(
            key="country",
            match=MatchAny(any=country),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if current_status:
        condition = FieldCondition(
            key="current_status",
            match=MatchAny(any=current_status),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if job_role:
        condition = FieldCondition(
            key="job_role",
            match=MatchAny(any=job_role),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if age_min is not None or age_max is not None:
        range_kwargs: dict = {}
        if age_min is not None:
            range_kwargs["gte"] = age_min
        if age_max is not None:
            range_kwargs["lte"] = age_max
        condition = FieldCondition(
            key="age",
            range=Range(**range_kwargs),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if financial_support_duration is not None:
        condition = FieldCondition(
            key="financial_support_duration",
            match=MatchValue(value=financial_support_duration),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    if financial_support_return:
        condition = FieldCondition(
            key="financial_support_return",
            match=MatchAny(any=financial_support_return),
        )
        field_must.append(condition)
        behaviour_must.append(condition)

    field_response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=field_query_vec,
        limit=top_k,
        query_filter=Filter(must=field_must),
    )

    behaviour_response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=behaviour_query_vec,
        limit=top_k,
        query_filter=Filter(must=behaviour_must),
    )

    field_results = field_response.points or []
    behaviour_results = behaviour_response.points or []

    students: dict[str, dict] = {}
    ranked_field: list[dict] = []
    ranked_behaviour: list[dict] = []

    for hit in field_results:
        payload = hit.payload or {}
        user_id = (
            payload.get("user_id")
            or payload.get("user")
            or payload.get("student_id")
        )
        if not user_id:
            continue
        score = hit.score or 0.0
        ranked_field.append(
            {
                "student_id": user_id,
                "user": user_id,
                "field_score": score,
                "summary_field": payload.get("summary_field"),
            }
        )

        entry = students.setdefault(
            user_id,
            {
                "student_id": user_id,
                "user": user_id,
                "field_score": 0.0,
                "behaviour_score": 0.0,
                "summary_field": payload.get("summary_field"),
                "summary_behaviour": None,
                "field_of_study": payload.get("field_of_study"),
                "country": payload.get("country"),
                "languages": payload.get("languages") or [],
                "current_status": payload.get("current_status"),
                "job_role": payload.get("job_role"),
                "financial_support_per_year": payload.get(
                    "financial_support_per_year"
                ),
                "financial_support_duration": payload.get(
                    "financial_support_duration"
                ),
                "financial_support_return": payload.get(
                    "financial_support_return"
                ),
            },
        )
        if score > (entry.get("field_score") or 0.0):
            entry["field_score"] = score
        if not entry.get("user"):
            entry["user"] = payload.get("user")
        if not entry.get("field_of_study") and payload.get("field_of_study"):
            entry["field_of_study"] = payload.get("field_of_study")
        if not entry.get("country") and payload.get("country"):
            entry["country"] = payload.get("country")
        if not entry.get("languages") and payload.get("languages"):
            entry["languages"] = payload.get("languages") or []

    for hit in behaviour_results:
        payload = hit.payload or {}
        user_id = (
            payload.get("user_id")
            or payload.get("user")
            or payload.get("student_id")
        )
        if not user_id:
            continue
        score = hit.score or 0.0
        ranked_behaviour.append(
            {
                "student_id": user_id,
                "user": user_id,
                "behaviour_score": score,
                "summary_behaviour": payload.get("summary_behaviour"),
            }
        )

        entry = students.setdefault(
            user_id,
            {
                "student_id": user_id,
                "user": user_id,
                "field_score": 0.0,
                "behaviour_score": 0.0,
                "summary_field": None,
                "summary_behaviour": payload.get("summary_behaviour"),
                "field_of_study": payload.get("field_of_study"),
                "country": payload.get("country"),
                "languages": payload.get("languages") or [],
                "current_status": payload.get("current_status"),
                "job_role": payload.get("job_role"),
                "financial_support_per_year": payload.get(
                    "financial_support_per_year"
                ),
                "financial_support_duration": payload.get(
                    "financial_support_duration"
                ),
                "financial_support_return": payload.get(
                    "financial_support_return"
                ),
            },
        )
        if score > (entry.get("behaviour_score") or 0.0):
            entry["behaviour_score"] = score
        if not entry.get("summary_behaviour"):
            entry["summary_behaviour"] = payload.get("summary_behaviour")
        if not entry.get("user"):
            entry["user"] = payload.get("user")
        if not entry.get("field_of_study") and payload.get("field_of_study"):
            entry["field_of_study"] = payload.get("field_of_study")
        if not entry.get("country") and payload.get("country"):
            entry["country"] = payload.get("country")
        if not entry.get("languages") and payload.get("languages"):
            entry["languages"] = payload.get("languages") or []

    temp_items: list[dict] = []
    for student_id, info in students.items():
        field_score_raw = info.get("field_score") or 0.0
        behaviour_score_raw = info.get("behaviour_score") or 0.0

        financial_score_raw = 0.0
        if financial_support_per_year is not None and financial_support_per_year > 0:
            student_need = info.get("financial_support_per_year")
            if isinstance(student_need, (int, float)) and student_need > 0:
                offer = float(financial_support_per_year)
                need = float(student_need)

                if offer >= need:
                    # Believer covers at least the full need: perfect raw match.
                    financial_score_raw = 1.0
                else:
                    # Believer offers less than the student's need.
                    # Use the ratio offer/need, but treat very low offers as 0.
                    ratio = offer / need  # in (0, 1)
                    if ratio <= 0.5:
                        # Offers less than half of the need: effectively no match.
                        financial_score_raw = 0.0
                    else:
                        # Linearly scale 0.5..1.0 -> 0..1.0
                        financial_score_raw = (ratio - 0.5) / 0.5

        temp_items.append(
            {
                "student_id": student_id,
                "user": info.get("user"),
                "field_score_raw": field_score_raw,
                "behaviour_score_raw": behaviour_score_raw,
                "financial_score_raw": financial_score_raw,
                "summary_field": info.get("summary_field"),
                "summary_behaviour": info.get("summary_behaviour"),
                "field_of_study": info.get("field_of_study"),
                "country": info.get("country"),
                "languages": info.get("languages") or [],
                "current_status": info.get("current_status"),
                "job_role": info.get("job_role"),
                "financial_support_per_year": info.get("financial_support_per_year"),
                "financial_support_duration": info.get("financial_support_duration"),
                "financial_support_return": info.get("financial_support_return"),
            }
        )

    # Use a single base value for field/behaviour normalization so that
    # the strongest signal across these two dimensions corresponds to 100%.
    base_raw = max(
        max((item["field_score_raw"] for item in temp_items), default=0.0),
        max((item["behaviour_score_raw"] for item in temp_items), default=0.0),
    )

    combined: list[dict] = []
    for item in temp_items:
        if base_raw > 0.0:
            field_score = item["field_score_raw"] / base_raw * 100.0
            behaviour_score = item["behaviour_score_raw"] / base_raw * 100.0
        else:
            field_score = 0.0
            behaviour_score = 0.0

        # Financial compatibility is already in [0, 1]; convert directly to %.
        financial_score = item["financial_score_raw"] * 100.0

        overall_match = (field_score + behaviour_score + financial_score) / 3.0

        combined.append(
            {
                "student_id": item["student_id"],
                "user": item.get("user"),
                "field_score": field_score,
                "behaviour_score": behaviour_score,
                "financial_score": financial_score,
                "overall_match": overall_match,
                "summary_field": item.get("summary_field"),
                "summary_behaviour": item.get("summary_behaviour"),
                "field_of_study": item.get("field_of_study"),
                "country": item.get("country"),
                "languages": item.get("languages") or [],
                "current_status": item.get("current_status"),
                "job_role": item.get("job_role"),
                "financial_support_per_year": item.get("financial_support_per_year"),
                "financial_support_duration": item.get("financial_support_duration"),
                "financial_support_return": item.get("financial_support_return"),
            }
        )

    if filter_languages:
        filter_langs = set(filter_languages)
        combined = [
            item
            for item in combined
            if filter_langs.intersection(
                set(students[item["student_id"]].get("languages") or [])
            )
        ]

    combined.sort(key=lambda x: x["overall_match"], reverse=True)

    return {
        "queries": queries,
        "field_results": ranked_field,
        "behaviour_results": ranked_behaviour,
        "combined_results": combined,
    }


class StudentSummaries(BaseModel):
    summary_field: str
    summary_behaviour: str


class StudentCreateRequest(BaseModel):
    user_id: str
    description: str
    field_of_study: str | None = None
    country: str | None = None
    languages: list[str] | None = None
    age: int | None = None
    current_status: str | None = None
    job_role: str | None = None
    financial_support_per_year: int | None = None
    financial_support_duration: int | None = None
    financial_support_return: str | None = None


class StudentCreateResponse(BaseModel):
    student_id: str
    user_id: str
    summaries: StudentSummaries


class BelieverMatchRequest(BaseModel):
    believer_text: str
    field_of_study: list[str] | None = None
    country: list[str] | None = None
    languages: list[str] | None = None
    current_status: list[str] | None = None
    job_role: list[str] | None = None
    age_min: int | None = None
    age_max: int | None = None
    financial_support_per_year: int | None = None
    financial_support_duration: int | None = None
    financial_support_return: list[str] | None = None


class MatchStudentResult(BaseModel):
    student_id: str
    user: str | None = None
    field_score: float
    behaviour_score: float
    financial_score: float
    overall_match: float
    summary_field: str | None = None
    summary_behaviour: str | None = None
    financial_support_per_year: int | None = None
    financial_support_duration: int | None = None
    field_of_study: str | None = None
    country: str | None = None
    languages: list[str] | None = None
    current_status: str | None = None
    job_role: str | None = None
    financial_support_return: str | None = None


class BelieverMatchResponse(BaseModel):
    queries: dict
    combined_results: list[MatchStudentResult]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/students", response_model=StudentCreateResponse)
def create_student(request: StudentCreateRequest) -> StudentCreateResponse:
    result = store_student_field_in_qdrant(
        student_description=request.description,
        user_id=request.user_id,
        field_of_study=request.field_of_study,
        country=request.country,
        languages=request.languages,
        age=request.age,
        current_status=request.current_status,
        job_role=request.job_role,
        financial_support_per_year=request.financial_support_per_year,
        financial_support_duration=request.financial_support_duration,
        financial_support_return=request.financial_support_return,
    )

    summaries = result.get("summaries", {})

    return StudentCreateResponse(
        student_id=result["student_id"],
        user_id=request.user_id,
        summaries=StudentSummaries(
            summary_field=summaries.get("summary_field", ""),
            summary_behaviour=summaries.get("summary_behaviour", ""),
        ),
    )


@app.post("/match-students", response_model=BelieverMatchResponse)
def match_students(request: BelieverMatchRequest) -> BelieverMatchResponse:
    result = match_students_for_believer(
        believer_text=request.believer_text,
        field_of_study=request.field_of_study,
        country=request.country,
        filter_languages=request.languages,
        current_status=request.current_status,
        job_role=request.job_role,
        age_min=request.age_min,
        age_max=request.age_max,
        financial_support_per_year=request.financial_support_per_year,
        financial_support_duration=request.financial_support_duration,
        financial_support_return=request.financial_support_return,
    )

    combined_results = [
        MatchStudentResult(
            student_id=item.get("student_id"),
            user=item.get("user"),
            field_score=item.get("field_score", 0.0),
            behaviour_score=item.get("behaviour_score", 0.0),
            financial_score=item.get("financial_score", 0.0),
            overall_match=item.get("overall_match", 0.0),
            summary_field=item.get("summary_field"),
            summary_behaviour=item.get("summary_behaviour"),
            financial_support_per_year=item.get("financial_support_per_year"),
            financial_support_duration=item.get("financial_support_duration"),
        )
        for item in result.get("combined_results", [])
    ]

    return BelieverMatchResponse(
        queries=result.get("queries", {}),
        combined_results=combined_results,
    )


if __name__ == "__main__":
    indexed_students: list[dict] = []

    # for student in MOCK_STUDENTS:
    #     res = store_student_field_in_qdrant(
    #         student_description=student["description"],
    #         user_id=student["user_id"],
    #     )
    #     indexed_students.append(
    #         {
    #             "user_id": student["user_id"],
    #             "student_id": res["student_id"],
    #             "summaries": res["summaries"],
    #         }
    #     )

    example_believer_text = (
        "We are looking for students with a strong background in computer science, "
        "data science, or software engineering, who are comfortable with Python and "
        "data-driven projects. In terms of behaviour, we value people who are "
        "reliable with deadlines, communicate clearly in remote teams, and can work "
        "independently while still being collaborative and open to feedback."
    )

    match_result = match_students_for_believer(example_believer_text, top_k=5)

    output = {
        "indexed_students": indexed_students,
        "match_result": match_result,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

import pandas as pd
import json
from llama_cpp import Llama
from typing import List
from pydantic import BaseModel, ConfigDict
from custom_dataclasses import ExtractedTable, ExtractionResult
import time

class Triplet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str

class TripletsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    triplets: List[Triplet]

schema = TripletsPayload.model_json_schema()


# SYSTEM_PROMPT = """
# Ты - модель для извлечения триплетов из CSV таблиц. 

# 1. subject — это название строки (левый заголовок, первый столбец).
# 2. object — это название колонки (верхний заголовок). Если пусто 
# 3. predicate — это значение в ячейке на пересечении этой строки и этой колонки.


# Пример таблицы:

# показатель,катушка,трансформатор,резерв, наименование способа реализации
# сила тока,1,2,3, с использованием промежуточных пунктов
# сопротивление,4,5,6, в стандартной форме

# Правильный результат:

# {
#   "triplets": [
#     {"subject": "сила тока", "predicate": "1", "object": "катушка"},
#     {"subject": "сила тока", "predicate": "2", "object": "трансформатор"},
#     {"subject": "сила тока", "predicate": "3", "object": "резерв"},
#     {"subject": "сила тока", "predicate": "с использованием промежуточных пунктов", "object": "наименование способа реализации"},
#     {"subject": "сопротивление", "predicate": "4", "object": "катушка"},
#     {"subject": "сопротивление", "predicate": "5", "object": "трансформатор"},
#     {"subject": "сопротивление", "predicate": "6", "object": "резерв"}
#     {"subject": "сопротивление", "predicate": "в стандартной форме", "object": "наименование способа реализации"}
#   ]
# }
# ]
# Если в клетке указан прочерк, то связи нет. Такое извлекать не нужно. Если верхний заголовок отсутствует оставь пустую строку.
# """

SYSTEM_PROMPT = """
Ты — модель для извлечения триплетов из CSV-таблиц.

Верни только JSON строго в формате:

{
  "triplets": [
    {"subject": "...", "predicate": "...", "object": "..."}
  ]
}

Правила извлечения:

1. Базовый случай: обычная матричная таблица
- subject — значение в первом столбце строки
- object — название колонки
- predicate — значение в ячейке на пересечении строки и колонки

Пример:
показатель,катушка,трансформатор
сила тока,1,2
сопротивление,4,5

Результат:
{
  "triplets": [
    {"subject": "сила тока", "predicate": "1", "object": "катушка"},
    {"subject": "сила тока", "predicate": "2", "object": "трансформатор"},
    {"subject": "сопротивление", "predicate": "4", "object": "катушка"},
    {"subject": "сопротивление", "predicate": "5", "object": "трансформатор"}
  ]
}

2. Если таблица состоит из одной строки заголовков и одной строки значений,
то это горизонтальная таблица параметр → значение.
В этом случае:
- subject — название колонки
- predicate — значение под этой колонкой
- object — пустая строка ""

Пример:
ток,сопротивление
1,2

Результат:
{
  "triplets": [
    {"subject": "ток", "predicate": "1", "object": ""},
    {"subject": "сопротивление", "predicate": "2", "object": ""}
  ]
}

3. Если таблица состоит из двух колонок, где в первом столбце указаны названия параметров,
а во втором столбце — их значения, то это вертикальная таблица параметр → значение.
В этом случае:
- subject — значение из первого столбца
- predicate — значение из второго столбца
- object — пустая строка ""

Пример:
показатель,
ток,1,2
сопротивление,2,0

Если CSV использует запятую как разделитель, то числа с запятой должны быть корректно восстановлены как одно значение.
Например 1,2 — это одно числовое значение, а не два разных столбца.

Результат:
{
  "triplets": [
    {"subject": "ток", "predicate": "1,2", "object": ""},
    {"subject": "сопротивление", "predicate": "2,0", "object": ""}
  ]
}

4. Если верхний заголовок отсутствует, пустой, технический или не несёт смысла,
оставь object равным пустой строке "".

5. Если в ячейке пусто, указан прочерк "-", "—", "–", null, NaN или значение отсутствует,
такую связь извлекать не нужно.

6. Не придумывай новые значения.
Извлекай только то, что явно есть в таблице.

7. Если таблица похожа на список параметров и значений, не пытайся интерпретировать её как матрицу.

8. Всегда сохраняй исходный текст значений как есть.
"""

model = Llama.from_pretrained(
    repo_id="t-tech/T-lite-it-2.1-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=True,
    n_ctx=4096
)

def table_to_csv_text(table: ExtractedTable, sep: str = ";") -> str:
    df = table.dataframe.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.map(
        lambda x: x.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        if isinstance(x, str) else x
    )
    csv_text = df.to_csv(index=False, sep=sep)
    return csv_text

def extraction_result_tables_to_csv(extraction_result: ExtractionResult, sep: str = ";"):
    csv_list_tables = []
    for table in extraction_result.tables:
        # table = extraction_result.tables[table_idx]
        csv_list_tables.append(table_to_csv_text(table, sep=sep))
    return csv_list_tables

def create_user_prompt(extraction_result):
    csv_list_tables = extraction_result_tables_to_csv(extraction_result)
    user_prompt = ""
    for i, csv_table in enumerate(csv_list_tables):
        user_prompt += f"Таблица {i+1}. {csv_table} \n"
    return user_prompt

def extract_triplets_by_llm(extraction_result):
    user_prompt = create_user_prompt(extraction_result)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    start = time.time()
    response = model.create_chat_completion(
        messages,
        response_format={
            "type": "json_object",
            "schema": schema
        },
        temperature=0.3,
        top_p=0.8,
        top_k=50,
        max_tokens=4096
    )
    end = time.time()
    print(f"ВРЕМЯ ВЫПОЛНЕНИЯ ЗАПРОСА {end-start}")
    print(f"ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС {user_prompt}")
    outputs = response["choices"][0]["message"]["content"]
    try:
        return json.loads(outputs)
    except:
        return outputs

    
import re 

try:
    import sqlglot

    def is_valid_sql(sql: str) -> bool:
        try:
            sqlglot.parse_one(sql)
            return True
        except Exception:
            return False

except ImportError:  # pragma: no cover - fail soft if sqlglot isn't installed
    def is_valid_sql(sql: str) -> bool:  # type: ignore[misc]
        return True


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL by removing extra whitespace and converting to lowercase.
    """
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.lower()

def normalize_exact_match(generated_sql: str , reference_sql:str)->bool:
    """
    Normalize SQL and check for exact match.
    """
    return normalize_sql(generated_sql) == normalize_sql(reference_sql)
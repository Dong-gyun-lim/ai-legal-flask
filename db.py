# db.py
import mariadb
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()  # .env 로드

@dataclass
class DBConfig:
    host: str = os.getenv("DB_HOST", "127.0.0.1")
    port: int = int(os.getenv("DB_PORT", "3306"))
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    database: str = os.getenv("DB_NAME", "")
    charset: str = "utf8mb4"
    autocommit: bool = True
    # 필요 시 SSL 비활성화용 (서버가 SSL 강제면 주석 해제하고 서버 설정에 맞춰 조정)
    ssl_disabled: bool = os.getenv("DB_SSL_DISABLED", "false").lower() == "true"

CFG = DBConfig()

def get_conn() -> mariadb.Connection:
    kwargs: Dict[str, Any] = dict(
        host=CFG.host,
        port=CFG.port,
        user=CFG.user,
        password=CFG.password,
        database=CFG.database,
        autocommit=CFG.autocommit,
    )
    # MariaDB Python 커넥터는 ssl=dict(...) 형태를 사용.
    # 필요할 때만 옵션 추가 (기본은 서버 설정 따름).
    if CFG.ssl_disabled:
        kwargs["ssl"] = {"disabled": True}

    conn = mariadb.connect(**kwargs)
    return conn

def fetch_one(sql: str, params: Optional[Tuple[Any, ...]] = None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return row
    finally:
        conn.close()

def fetch_all(sql: str, params: Optional[Tuple[Any, ...]] = None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()

def test_connection() -> None:
    """
    연결 확인 + precedent_chunks 개수 점검
    """
    print(f"[DB] host={CFG.host} db={CFG.database} user={CFG.user}")
    # 1) 간단 핑: 버전
    row = fetch_one("SELECT VERSION()")
    print(f"[DB] version: {row[0] if row else 'unknown'}")

    # 2) 청크 테이블 카운트 (있으면 개수, 없으면 에러 메시지)
    try:
        r = fetch_one("SELECT COUNT(*) FROM precedent_chunks")
        print(f"[DB] precedent_chunks count: {r[0]}")
    except mariadb.Error as e:
        print("[DB] precedent_chunks 점검 실패:", e)

if __name__ == "__main__":
    test_connection()

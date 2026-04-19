import os
import zipfile
import requests
from dotenv import load_dotenv

COMPETITION = "home-credit-default-risk"
BASE_URL = "https://www.kaggle.com/api/v1"


def get_auth_headers():
    load_dotenv()
    token = os.getenv("KAGGLE_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    raise EnvironmentError("Configure KAGGLE_API_TOKEN no .env")


def list_competition_files(competition: str, headers: dict):
    url = f"{BASE_URL}/competitions/data/list/{competition}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    return data.get("files", data) if isinstance(data, dict) else data


def download_file(competition: str, filename: str, dest_dir: str, headers: dict):
    """Baixa um arquivo individual usando a URL de download direto do Kaggle."""
    url = f"{BASE_URL}/competitions/data/download/{competition}/{filename}"
    print(f"  Baixando {filename}...", end=" ", flush=True)

    # stream=True + allow_redirects para seguir o redirect até o GCS
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        dest_path = os.path.join(dest_dir, filename)
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

    size_mb = downloaded / 1e6
    print(f"✓ ({size_mb:.1f} MB)")
    return dest_path


def extract_if_zip(path: str):
    """Verifica se é ZIP real e extrai no mesmo diretório."""
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic[:2] == b"PK":
        dest_dir = os.path.dirname(path)
        print(f"    → Extraindo {os.path.basename(path)}...", end=" ")
        try:
            with zipfile.ZipFile(path) as zf:
                inner_names = zf.namelist()
                zf.extractall(dest_dir)
            # Remove o ZIP e verifica se o CSV foi extraído com o mesmo nome
            os.remove(path)
            for inner_name in inner_names:
                extracted = os.path.join(dest_dir, inner_name)
                csv_path = path.replace(".zip", "") if path.endswith(".zip") else path
                if os.path.exists(extracted) and extracted != path:
                    # Se o conteúdo extraído é o arquivo que queremos, nada a fazer
                    pass
                print(f"OK (extraído: {inner_name})")
        except zipfile.BadZipFile:
            print("⚠️  Arquivo ZIP corrompido — mantendo original")
    else:
        print(f"    → Arquivo plain text, sem extração necessária")


def main():
    headers = get_auth_headers()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Listando arquivos da competição '{COMPETITION}'...")
    files = list_competition_files(COMPETITION, headers)
    print(f"Encontrados {len(files)} arquivo(s).\n")

    for f in files:
        name = f["name"]
        dest_path = os.path.join(DATA_DIR, name)

        # Pula arquivos já existentes e corretos (> 1KB)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            # Verifica se não é um ZIP truncado
            with open(dest_path, "rb") as fp:
                magic = fp.read(4)
            if magic[:2] == b"PK":
                print(f"  {name} — ZIP detectado, será re-baixado e extraído")
            else:
                size_mb = os.path.getsize(dest_path) / 1e6
                print(f"  {name} — já existe ({size_mb:.1f} MB), pulando")
                continue

        path = download_file(COMPETITION, name, DATA_DIR, headers)
        extract_if_zip(path)

    print("\nConcluído! Arquivos em data/:")
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith(".csv") or fn.endswith(".zip"):
            size_mb = os.path.getsize(os.path.join(DATA_DIR, fn)) / 1e6
            print(f"  {fn:50s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()

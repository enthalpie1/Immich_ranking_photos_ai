import io, json, os
import numpy as np
import requests
import torch
import torch.nn as nn
import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

def charger_env(chemin):
    for ligne in Path(chemin).expanduser().read_text().splitlines():
        ligne = ligne.strip()
        if ligne and not ligne.startswith("#") and "=" in ligne:
            cle, valeur = ligne.split("=", 1)
            os.environ.setdefault(cle.strip(), valeur.strip().strip('"').strip("'"))

charger_env("~/config/immich-ranking.env")
API_KEY = os.environ["IMMICH_API_KEY"]
URL = os.environ["IMMICH_URL"]

# ---------- Réglages ----------
ALBUM_ID = "b85f6c2b-8e36-419a-9969-9af44a072339"
DRY_RUN = True          # True = n'écrit rien dans Immich
W_SHARP = 0.3           # poids de la netteté face à l'esthétique (0 = esthétique seule)
CACHE_FILE = "scores_laion.json"
WEIGHTS_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
               "/raw/main/sac+logos+ava1-l14-linearMSE.pth")
WEIGHTS_FILE = "sac+logos+ava1-l14-linearMSE.pth"
# ------------------------------

H = {"x-api-key": API_KEY}
device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    """Même architecture que le prédicteur LAION (les noms de couches doivent
    correspondre au fichier de poids)."""
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def charger_modeles():
    if not os.path.exists(WEIGHTS_FILE):
        print("Téléchargement des poids du prédicteur...")
        r = requests.get(WEIGHTS_URL, timeout=120)
        r.raise_for_status()
        with open(WEIGHTS_FILE, "wb") as f:
            f.write(r.content)
    mlp = MLP()
    mlp.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
    mlp.to(device).eval()
    print("Chargement de CLIP (1,7 Go au premier lancement)...")
    nom = "openai/clip-vit-large-patch14"
    clip = CLIPModel.from_pretrained(nom).to(device).eval()
    proc = CLIPProcessor.from_pretrained(nom)
    return mlp, clip, proc


@torch.no_grad()
def score_esthetique(img, mlp, clip, proc):
    entrees = proc(images=img, return_tensors="pt").to(device)
    emb = clip.get_image_features(**entrees)
    emb = emb / emb.norm(dim=-1, keepdim=True)      # normalisation L2, indispensable
    return mlp(emb.float()).item()


def netteté(img):
    gris = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return float(np.log1p(cv2.Laplacian(gris, cv2.CV_64F).var()))


def album_images(album_id):
    images, page = [], 1
    while page:
        r = requests.post(
            f"{URL}/search/metadata", headers=H,
            json={"albumIds": [album_id], "type": "IMAGE",
                  "page": page, "size": 250},
        )
        r.raise_for_status()
        bloc = r.json()["assets"]
        images.extend(bloc["items"])
        page = int(bloc["nextPage"]) if bloc.get("nextPage") else None
    return images


def main():
    assets = album_images(ALBUM_ID)
    noms = {a["id"]: a.get("originalFileName", a["id"]) for a in assets}
    print(f"{len(assets)} photos dans l'album (device : {device})")

    cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
    a_calculer = [a for a in assets if a["id"] not in cache]

    if a_calculer:
        mlp, clip, proc = charger_modeles()
        for i, a in enumerate(a_calculer, 1):
            r = requests.get(f"{URL}/assets/{a['id']}/thumbnail",
                             headers=H, params={"size": "preview"})
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            cache[a["id"]] = {"aesth": score_esthetique(img, mlp, clip, proc),
                              "sharp": netteté(img)}
            print(f"\r{i}/{len(a_calculer)}", end="", flush=True)
            if i % 25 == 0:
                json.dump(cache, open(CACHE_FILE, "w"))
        print()
        json.dump(cache, open(CACHE_FILE, "w"))

    ids = [a["id"] for a in assets if a["id"] in cache]
    aesth = np.array([cache[i]["aesth"] for i in ids])
    sharp = np.array([cache[i]["sharp"] for i in ids])

    def z(x):
        return (x - x.mean()) / (x.std() or 1)

    score = z(aesth) + W_SHARP * z(sharp)
    ordre = np.argsort(-score)
    n = len(ordre)

    print("\nTop 5 :")
    for idx in ordre[:5]:
        print(f"  {noms[ids[idx]]}  (esthétique {aesth[idx]:.2f}, score {score[idx]:.2f})")
    print("Derniers 5 :")
    for idx in ordre[-5:]:
        print(f"  {noms[ids[idx]]}  (esthétique {aesth[idx]:.2f}, score {score[idx]:.2f})")

    paliers = [(0.10, 5), (0.25, 4), (0.50, 3)]    # top 10 %, 25 %, 50 % (cumulés)
    for rang, idx in enumerate(ordre):
        part = (rang + 1) / n
        note = next((s for limite, s in paliers if part <= limite), None)
        if note is None:
            continue
        if not DRY_RUN:
            requests.put(f"{URL}/assets/{ids[idx]}", headers=H,
                         json={"rating": note}).raise_for_status()

    print("\nTerminé" + (" (simulation, rien n'a été écrit)" if DRY_RUN else ""))


if __name__ == "__main__":
    main()
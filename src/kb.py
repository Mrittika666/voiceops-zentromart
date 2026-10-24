import glob, os, re
import numpy as np, faiss
from sentence_transformers import SentenceTransformer


def _parse(path):
    text = open(path, encoding="utf-8").read()
    meta, body = {}, text
    m = re.match(r"\s*(?:---\s*\n)?(.*?)\n---\s*\n(.*)", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


class KB:
    def __init__(self, kb_dir):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
            fname = os.path.basename(path)
            meta, body = _parse(path)
            for i, p in enumerate(re.split(r"(?m)^(?=#{2,3} )", body)):
                p = p.strip()
                if len(p) < 30:
                    continue
                sid = re.search(r"\[([A-Za-z0-9\-_]+)\]", p.splitlines()[0])
                self.chunks.append({
                    "file": fname,
                    "section_id": sid.group(1) if sid else f"{fname}#{i}",
                    "title": meta.get("title", ""),
                    "status": meta.get("status", ""),
                    "effective_from": meta.get("effective_from", ""),
                    "text": p,
                })
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        texts = [f'{c["title"]} [{c["section_id"]}]\n{c["text"]}' for c in self.chunks]
        vecs = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.esc = [c for c in self.chunks if c["file"] == "agent_escalation_handoff.md"]

    def retrieve(self, query, k=5):
        q2, ql = query, query.lower()
        if re.search(r"refund|paisa|पैसे|रिफ", ql):
            q2 += " refund timeline by payment method business days"
        if re.search(r"return|wapas|वापस|replace", ql):
            q2 += " return window delivery date return fee reverse pickup"
        if re.search(r"damage|broken|kharab|टूट|wrong|missing", ql):
            q2 += " damaged wrong missing item claim"
        qv = self.model.encode([q2], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(qv, 20)
        out = [self.chunks[i] for i in idx[0] if self.chunks[i]["status"] != "superseded"][:k]
        seen = {(c["file"], c["section_id"]) for c in out}
        return out + [c for c in self.esc if (c["file"], c["section_id"]) not in seen]

import argparse, csv, glob, json, os, re, sys, time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("--out", default="sample_outputs")
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR", "Capstone_Project_Dataset_FTTT_2026"))
    a = ap.parse_args()

    data = os.path.abspath(a.data)
    sys.path.insert(0, data)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.kb import KB
    from src.agent import Agent, build_system
    from src import voice

    kb = KB(os.path.join(data, "kb"))
    system = build_system(data)
    os.makedirs(a.out, exist_ok=True)

    exts = ("wav", "mp3", "m4a", "flac", "ogg")
    files = sorted([f for e in exts for f in glob.glob(os.path.join(a.audio_dir, f"*.{e}"))], key=natural)
    pat = re.compile(r"^([A-Za-z]+\d+)[_-]t?(\d+)$")
    agents, rows = {}, []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        m = pat.match(stem)
        key = m.group(1) if m else stem
        if key not in agents:
            agents[key] = Agent(kb, system)
        ag = agents[key]

        heard = voice.listen(f)
        if not heard:
            reply, chunks_used, tools = voice.NO_AUDIO, [], []
        else:
            reply = ag.respond(heard)
            chunks_used, tools = ag.last_chunks, ag.tool_log

        out_wav = os.path.join(a.out, f"{stem}_reply.wav")
        try:
            voice.speak(voice.for_ear(reply), out_wav)
        except Exception as e:
            out_wav = f"TTS_ERROR: {str(e)[:150]}"

        rows.append({"file": os.path.basename(f), "heard": heard, "reply": reply,
                     "chunks_used": " ; ".join(chunks_used), "tool_calls": " ; ".join(tools),
                     "reply_audio": out_wav})
        print(f"[{stem}] heard: {heard[:80]!r} -> {reply[:100]!r}")
        time.sleep(float(os.environ.get("CALL_GAP", "2")))

    json.dump(rows, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if rows:
        with open(os.path.join(a.out, "results.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("Done:", len(rows), "files ->", a.out)


if __name__ == "__main__":
    main()

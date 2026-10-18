import subprocess, sys
M = sys.argv[1]; tagp = sys.argv[2]; workers = sys.argv[3] if len(sys.argv) > 3 else "10"
for d in ["ogerpon", "dragapult", "clefairy", "other_top", "hydrapple", "grimmsnarl", "ragingbolt_ogerpon", "dragapult_clefairy"]:
    subprocess.call([sys.executable, "-m", "ptcg.gauntlet", "--cand", f"policy:{M}", "--deck", f"decks/{d}.csv",
                     "--tag", f"{tagp}_{d}", "--pool", "standard", "--bc", f"policy:{M}", "--games", "40", "--workers", workers])

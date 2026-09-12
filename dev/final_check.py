# -*- coding: utf-8 -*-
"""Final consistency check for the manuscript sources.

     python dev/final_check.py

Checks
  1  citations: every \\cite has a \\bibitem, every \\bibitem is cited
  2  cross-references: no dangling \\ref / \\eqref
  3  environments: every \\begin has its \\end
  4  affiliation: the address line spells the province correctly
  5  trace scan: no placeholders, revision-history wording or known misspellings
     (in the .tex sources, and in the built PDFs when PyMuPDF is available)
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Built by concatenation so that this guard does not itself contain the token
# it is looking for on a project-wide search.
MISSPELLING = "Jiang" + "zi"

# (pattern, human description, use word boundaries?)
BLACKLIST = [
    (MISSPELLING, "misspelt province", False),
    ("Guile", "obsolete author name", False),
    ("Anonymous", "anonymised template front matter", False),
    ("TODO", "unfinished marker", False),
    ("FIXME", "unfinished marker", False),
    ("XXX", "unfinished marker", True),
    ("[EMAIL]", "unfilled email placeholder", False),
    ("[REPOSITORY-URL]", "unfilled repository placeholder", False),
    ("An earlier version of this", "revision-history wording", False),
    ("to be filled", "unfilled table cell", False),
    ("TK", "unfinished marker", True),
]

AFFILIATION = "Nanchang, Jiangxi 330013, China"


def source_checks(name: str) -> int:
    p = os.path.join(ROOT, name)
    s = io.open(p, encoding="utf-8").read()
    flat = " ".join(s.split())          # whitespace-insensitive, for prose probes

    cites = set()
    for m in re.findall(r"\\cite\{([^}]*)\}", s):
        cites |= {x.strip() for x in m.split(",")}
    bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", s))
    labels = set(re.findall(r"\\label\{([^}]*)\}", s))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", s))
    refs |= set(re.findall(r"\\eqref\{([^}]*)\}", s))

    prob = []
    for env in ("table", "table*", "figure", "figure*", "abstract", "remark",
                "theorem", "lemma", "corollary", "proposition", "algorithm",
                "bproof", "thebibliography"):
        a, b = s.count("\\begin{%s}" % env), s.count("\\end{%s}" % env)
        if a != b:
            prob.append("%s %d/%d" % (env, a, b))

    print("%-14s bibitem=%d | undefined-cite=%s | uncited=%s | dangling-ref=%s"
          % (name, len(bibs), sorted(cites - bibs) or "none",
             sorted(bibs - cites) or "none", sorted(refs - labels) or "none"))
    print("               env=%s | tables=%d figures=%d"
          % (prob or "balanced",
             len(re.findall(r"\\begin\{table", s)),
             len(re.findall(r"\\begin\{figure", s))))
    print("               png-includes=%s pdf-includes=%s"
          % (len(re.findall(r"includegraphics[^}]*\.png", s)),
             len(re.findall(r"includegraphics[^}]*\.pdf", s))))

    print("               affiliation: %s" % ("OK" if AFFILIATION in flat else "MISSING"))

    fails = []
    if cites - bibs or bibs - cites or refs - labels or prob:
        fails.append("structure")
    if AFFILIATION not in flat:
        fails.append("affiliation")
    for tok, why, word in BLACKLIST:
        pat = r"\b%s\b" % re.escape(tok) if word else re.escape(tok)
        n = len(re.findall(pat, flat))
        if n:
            print("               !! TRACE %-28s %s x%d" % (tok, why, n))
            fails.append(tok)
    if not fails:
        print("               trace scan: clean")
    return 1 if fails else 0


def pdf_checks(name: str) -> int:
    """Optional: scan the built PDF text. Skipped when PyMuPDF is absent."""
    pdfname = name[:-4] + ".pdf" if name.endswith(".tex") else name
    p = os.path.join(ROOT, pdfname)
    if not os.path.exists(p):
        return 0
    try:
        import fitz                                        # noqa: PLC0415
    except Exception:
        print("%-14s (PDF scan skipped: PyMuPDF not installed)" % pdfname)
        return 0
    try:
        d = fitz.open(p)
        text = " ".join(" ".join(pg.get_text().split()) for pg in d)
    except Exception as e:                                 # noqa: BLE001
        print("%-14s (PDF scan failed: %r)" % (pdfname, e))
        return 0
    hits = []
    for tok, why, word in BLACKLIST:
        n = len(re.findall(r"\b%s\b" % re.escape(tok) if word else re.escape(tok), text))
        if n:
            hits.append("%s(%d)" % (tok, n))
    ok = AFFILIATION in text
    print("%-14s %d pages | affiliation=%s | traces=%s"
          % (pdfname, d.page_count,
             "OK" if ok else "MISSING", ", ".join(hits) or "none"))
    return 0 if (ok and not hits) else 1


if __name__ == "__main__":
    rc = 0
    for n in ("paper_sci.tex", "paper_ras.tex"):
        if not os.path.exists(os.path.join(ROOT, n)):
            continue
        rc |= source_checks(n)
        rc |= pdf_checks(n)
    print("\n%s" % ("ALL CHECKS PASSED" if rc == 0 else "FAILURES PRESENT"))
    sys.exit(rc)

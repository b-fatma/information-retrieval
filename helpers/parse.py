def parse_medline_docs(path):
    docs = {}
    doc_id = None
    buffer = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(".I"):
                if doc_id is not None:
                    docs[doc_id] = " ".join(buffer)
                doc_id = int(line.split()[1])
                buffer = []

            elif line.startswith(".W"):
                continue

            else:
                buffer.append(line)

        if doc_id is not None:
            docs[doc_id] = " ".join(buffer)

    return docs


def parse_medline_queries(path):
    queries = {}
    qid = None
    buffer = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(".I"):
                if qid is not None:
                    queries[qid] = " ".join(buffer)
                qid = int(line.split()[1])
                buffer = []

            elif line.startswith(".W"):
                continue

            else:
                buffer.append(line)

        if qid is not None:
            queries[qid] = " ".join(buffer)

    return queries


def parse_medline_rel(path):
    rels = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                continue  # skip malformed lines

            qid = int(parts[0])
            doc_id = int(parts[2])

            if qid not in rels:
                rels[qid] = set()
            rels[qid].add(doc_id)

    return rels

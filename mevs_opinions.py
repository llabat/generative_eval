import json

with open("/home/labat/polyglots_or_multitudes/mevs/data/MEVS.json", "r") as f:
    mevs = json.load(f)

def get_statement(qid):
    return mevs["translations"][mevs["questions"][str(qid)]["requests"][-1]]["ENG_GB"]

TEST_QUESTIONS = [13, 14, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 118, 119, 120, 121, 122]
mevs_opinions = [(q, get_statement(q) + ".") for q in TEST_QUESTIONS]

GOOD_OR_BAD = [(q, get_statement(q)) for q in {86, 87, 88, 89}]
suffix = " is best for the country"
gob_opinion = [(q, statement + suffix) if not statement.endswith(suffix) else statement for q, statement in GOOD_OR_BAD]

GOVERMENT_TEMPLATE = "The government should have the right to {measure}."
gov_statements = [(q, get_statement(q)) for q in {124, 125, 126}]
gov_opinions = [(q, GOVERMENT_TEMPLATE.format(measure = gstat[0].lower() + gstat[1:])) for q, gstat in gov_statements]

altruism_opinions = [(q, "{opinion} is important.".format(opinion = get_statement(q))) for q in {136, 137, 138}]

OPINIONS = mevs_opinions + gov_opinions + altruism_opinions
ID2OPINION = dict(OPINIONS)
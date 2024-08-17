from config import db

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(30), unique=False, nullable=False)
    last_name = db.Column(db.String(30), unique=False, nullable=False)
    party = db.Column(db.String(30), unique=False, nullable=False)
    election_year = db.Column(db.Integer, unique=False, nullable=False)
    web_link = db.Column(db.String(50), unique=True, nullable=False)
    defense = db.Column(db.String(500), unique=True, nullable=True)
    defense = db.Column(db.String(500), unique=True, nullable=True)
    democracy = db.Column(db.String(500), unique=True, nullable=True)
    economy = db.Column(db.String(500), unique=True, nullable=True)
    education = db.Column(db.String(500), unique=True, nullable=True)
    environment = db.Column(db.String(500), unique=True, nullable=True)
    foreign_policy = db.Column(db.String(500), unique=True, nullable=True)
    healthcare = db.Column(db.String(500), unique=True, nullable=True)
    immigration = db.Column(db.String(500), unique=True, nullable=True)
    infrastructure = db.Column(db.String(500), unique=True, nullable=True)
    social_issues = db.Column(db.String(500), unique=True, nullable=True)
    tech = db.Column(db.String(500), unique=True, nullable=True)

    def to_json(self):
        return {
            "id": self.id,
            "firstName": self.first_name,
            "lastName": self.last_name,
            "party": self.party,
            "election_year": self.election_year,
            "webLink": self.web_link,
            "democracy": self.democracy,
            "economy": self.economy,
            "education": self.education,
            "environment": self.environment,
            "foreignPolicy": self.foreign_policy,
            "healthcare": self.healthcare,
            "immigration": self.immigration,
            "infrastructure": self.infrastructure,
            "socialIssues": self.social_issues,
            "tech": self.tech
        }

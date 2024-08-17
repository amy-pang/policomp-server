from flask import request, jsonify
from config import app, db
from models import Candidate

# Retrieve all candidates and their respective information
@app.route("/candidates", methods=["GET"])
def get_candidates():
    candidates = Candidate.query.all()
    json_candidates = list(map(lambda x: x.to_json(), candidates))
    return jsonify({"candidates": json_candidates})

# Retrieve most recent candidate by party
@app.route("/recent_candidate", methods=["GET"])
def get_recent_candidate():
    # Get the party from the query parameters
    party = request.args.get('party')
    
    if not party:
        return jsonify({"error": "Party parameter is required"}), 400

    # Subquery to find the most recent election year for the specified party
    subquery = db.session.query(
        Candidate.party,
        db.func.max(Candidate.election_year).label('recent_year')
    ).filter(Candidate.party == party).group_by(Candidate.party).subquery()

    # Join with the main Candidate table to get the most recent candidate for the specified party
    candidate = db.session.query(Candidate).join(
        subquery,
        (Candidate.party == subquery.c.party) & (Candidate.election_year == subquery.c.recent_year)
    ).first()

    if not candidate:
        return jsonify({"error": "No candidate found for the specified party"}), 404

    # Convert to JSON
    return jsonify({"candidate": candidate.to_json()})


"""
# Create a candidate
@app.route("/create_candidate", methods=["POST"])
def create_candidate():
    first_name = request.json.get("firstName")
    last_name = request.json.get("lastName")
    web_link = request.json.get("webLink")

    if not first_name or not last_name or not web_link:
        return (
            jsonify({"message": "You must include a first name, last name, and website link."}),
            400,
        )

    new_candidate = Candidate(first_name=first_name, last_name=last_name, web_link=web_link)
    try:
        db.session.add(new_candidate)
        db.session.commit()
    except Exception as e:
        return jsonify({"message": str(e)}), 400

    return jsonify({"message": "Candidate created!"}), 201


@app.route("/update_candidate/<int:candidate_id>", methods=["PATCH"])
def update_candidate(candidate_id):
    candidate = Candidate.query.get(candidate_id)

    if not candidate:
        return jsonify({"message": "Candidate not found"}), 404

    data = request.json
    candidate.first_name = data.get("firstName", candidate.first_name)
    candidate.last_name = data.get("lastName", candidate.last_name)
    candidate.web_link = data.get("webLink", candidate.web_link)

    db.session.commit()

    return jsonify({"message": "Candidate updated."}), 200


@app.route("/delete_contact/<int:user_id>", methods=["DELETE"])
def delete_contact(user_id):
    contact = Contact.query.get(user_id)

    if not contact:
        return jsonify({"message": "User not found"}), 404

    db.session.delete(contact)
    db.session.commit()

    return jsonify({"message": "User deleted!"}), 200
"""


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run(debug=True)

'''
@app.before_first_request
def create_tables():
    db.create_all()
'''

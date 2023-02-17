import json, requests
from datetime import datetime,timedelta

from flask import Flask, render_template, session, request, redirect, flash
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///aitrainer.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.secret_key = "0x29874909298974790097629842"
app.permanent_session_lifetime = timedelta(minutes=30)

db = SQLAlchemy(app) 
bootstrap = Bootstrap(app)

@app.route("/", methods=["GET"])
def list():
    if "flag" in session and session["flag"]:
        msg = str(session["uid"]) + "'s records"
        data = Record.query.filter_by(uid=session["uid"]).all()
        return render_template("list.html",
                               title="Records",
                               message=msg,
                               data=data)
    else:
        return redirect("/login")
    
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html",
                                title="A login page")
    uid = request.form["uid"]
    pwd = request.form["pwd"]
    user = db.session.query(User).filter_by(uid=uid).first()
    if user:
        if db.session.query(User).filter_by(password=pwd).first():
            session["flag"] = True
        
        else:
            session["flag"] = False
    
    else:
        session["flag"] = False
        return render_template("login.html",
                               title="A login page",
                               message_id="User ID is wrong")
        
    session["uid"] = uid
    
    if session["flag"]:
        return redirect("/")
    else:
        return render_template("login.html",
                               title="A login page",
                               message_pwd="Password is wrong")
    
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("uid", None)
    session.pop("flag", None)
    return redirect("/")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html",
                                title="A signup page")
    
    uid = request.form["uid"]
    password = request.form["pwd"]
    user = db.session.query(User).filter_by(uid=uid).first()
    if user:
        return render_template("signup.html",
                               title="A signup page",
                               message_id="This user id is already used")
    
    new_uid = User(uid=uid, password=password)
    db.session.add(new_uid)
    db.session.commit()
    return redirect("/")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "GET":
        if "flag" in session and session["flag"]:
            return render_template("admin.html",
                                    title="A admin page")
        else:
            return render_template("login.html",
                                    title="A login page")
    
    uid = request.form["uid"]
    method = request.form["method"]
    count = request.form["count"]
    
    if method == "":
        return render_template("admin.html",
                               title="A admin page",
                               message_method="Select a method")
    
    if count == "":
        return render_template("admin.html",
                               title="A admin page",
                               message_count="Select reps you did")
    
    if db.session.query(User).filter_by(uid=uid).first():
        new_record = Record(uid=uid, method=method, count=count)
        db.session.add(new_record)
        db.session.commit()
        return redirect("/")
    
    else:
        return render_template("admin.html",
                               title="A admin page",
                               message_id="User ID is wrong")
    
@app.route("/run-camera", methods=["POST"])
def run_camera():
    # response = requests.get('http://<jetson-ip>:<port>/run-code')
    # if response.ok:
    if True:
        flash("Running", "alert-success")
    else:
        flash("Error", "alert-danger")
    return redirect("/")

@app.route("/fetch-data", methods=["POST"])
def fetch_data():
    try:
        json_data = json.loads(request.data, strict=False)
        return "JSON data loaded successfully"
    except ValueError as e:
        return "Invalid JSON: {}".format(str(e))
    except requests.exceptions.HTTPError as e:
        return "HTTP error: {}".format(str(e))
    
    #---- requires more error handlings and redirecting settings to the codes above
    
    
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(255))
    password = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

class Record(db.Model):
    __tablename__ = "record"
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(255))
    method = db.Column(db.String(255))
    count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

if __name__ == "__main__":
    app.run(debug=True, port=3030)


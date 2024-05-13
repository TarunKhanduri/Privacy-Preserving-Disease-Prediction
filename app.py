from flask import Flask, render_template, request,url_for
import logic
app= Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template("index.html")

# move to heart disease page
@app.route('/disease', methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        page="heart.html"
    return render_template(page)

# heart disease analysis
@app.route('/analyse_h', methods=["GET", "POST"])
def analyse_h():
    if request.method == "POST":
        input=[]
        for i in range(1,14):
            input.append(float(request.form["input"+str(i)]))
        m="Models/heart_model.pkl"
        n="Models/heart_scaler.pkl"
        
        try:
            output_svm = logic.predict_heart(m, n, input)
        except Exception as e:
            output_svm = str(e)
        
        try:
            output_ckks = logic.predict_heart_fhe(m, n, input)
        except Exception as e:
            output_ckks = str(e)
        
        try:
            output_ckks_dp = logic.predict_heart_fhe_dp(m, n, input)
        except Exception as e:
            output_ckks_dp = str(e)
        
        try:
            output_paillier = logic.predict_heart_phe(m, n, input)
        except Exception as e:
            output_paillier = str(e)
        
        try:
            output_dp = logic.predict_heart_dp(m, n, input)
        except Exception as e:
            output_dp = str(e)


        return render_template("predict.html",opt1= output_svm, opt2=output_ckks, opt3= output_ckks_dp, opt4= output_paillier, opt5= output_dp)
    return render_template("index.html")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/back", methods=["GET", "POST"])
def back():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

from asyncio.base_futures import _format_callbacks
from email.mime import image
import imp
from flask import Flask, Request,render_template




app = Flask(__name__)

@ app.route('/home')
def home():

return render_template("home.html")


@app.route('/')
def getimage():

    return image
@app.rout('/load')
def uploadimage():

    print("Image uploaded successfully")
   

if __name__ == '__main__':
    app.run()
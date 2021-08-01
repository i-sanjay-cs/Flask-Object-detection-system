from flask import Flask, render_template, Response
from camera import VideoLive,Face


app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/page1/')
def page1():
    return render_template('page1.html')
def obdetect(camera):
    while True:
        frame=camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(obdetect(VideoLive()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/page2/')
def page2():
    return render_template('page2.html')
def facedetect(camera):
    while True:
        frame2=camera.get_face()
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')

@app.route('/video_feed1')
def video_feed1():
    return Response(facedetect(Face()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
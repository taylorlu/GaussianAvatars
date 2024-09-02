from flask import Flask, send_from_directory, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all domains to access this server

FILE_DIRECTORY = '.'

@app.route('/<path:filename>', methods=['GET'])
def get_file(filename):
    try:
        # 安全检查，确保不能访问到上层目录
        if '..' in filename or filename.startswith('/'):
            abort(400, "Invalid file name")

        # 返回文件
        return send_from_directory(FILE_DIRECTORY, filename)
    except FileNotFoundError:
        abort(404, "File not found")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
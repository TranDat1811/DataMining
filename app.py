#thêm các thư viện
import numpy as np
from flask import Flask, request, render_template, Response
import pickle
import pandas as pd


#khai báo flask
app = Flask(__name__)

#đọc mô hình đã xây dựng được từ file model/test.py
model = pickle.load(open('divorceModelComplex.pkl', 'rb'))

#đọc các câu hỏi
questions = ["Nếu một người trong chúng ta xin lỗi khi cuộc thảo luận của chúng ta xấu đi, cuộc thảo luận sẽ kết thúc.",
             "Tôi biết chúng ta có thể bỏ qua sự khác biệt của mình, ngay cả khi đôi khi mọi thứ trở nên khó khăn.",
             "Khi cần, chúng tôi có thể thảo luận với vợ / chồng tôi ngay từ đầu và sửa lại.",
             "Khi tôi thảo luận với vợ / chồng của mình, để liên lạc với anh ấy cuối cùng sẽ hiệu quả.",
             "Khoảng thời gian tôi ở bên vợ là đặc biệt đối với chúng tôi.",
             "Chúng tôi không có thời gian ở nhà với tư cách là đối tác.",
             "Chúng tôi giống như hai người xa lạ cùng chung môi trường ở nhà hơn là gia đình.",
             "Tôi tận hưởng kỳ nghỉ của chúng tôi với vợ tôi.",
             "Tôi thích đi du lịch với vợ tôi.",
             "Hầu hết các mục tiêu của chúng tôi là chung cho vợ / chồng tôi.",
             "Tôi nghĩ rằng một ngày nào đó trong tương lai, khi nhìn lại, tôi thấy vợ chồng tôi đã rất hòa hợp "
             "với nhau.",
             "Vợ / chồng của tôi và tôi có những giá trị tương tự nhau về quyền tự do cá nhân.",
             "Vợ / chồng của tôi và tôi có cảm giác giải trí giống nhau.",
             "Hầu hết các mục tiêu của chúng tôi đối với mọi người (trẻ em, bạn bè, v.v.) đều giống nhau.",
             "Những giấc mơ của chúng tôi với người phối ngẫu của tôi rất giống nhau và hài hòa.",
             "Chúng tôi tương thích với người phối ngẫu của tôi về tình yêu nên là gì.",
             "Chúng tôi có cùng quan điểm về hạnh phúc trong cuộc sống với người bạn đời của tôi",
             "Vợ / chồng của tôi và tôi có những ý tưởng giống nhau về cách kết hôn",
             "Vợ / chồng của tôi và tôi có những ý tưởng giống nhau về vai trò nên có trong hôn nhân như thế nào",
             "Vợ / chồng của tôi và tôi có những giá trị tương tự về sự tin tưởng.",
             "Tôi biết chính xác những gì vợ tôi thích.",
             "Tôi biết vợ / chồng tôi muốn được chăm sóc như thế nào khi cô ấy / anh ấy ốm.",
             "Tôi biết món ăn yêu thích của vợ / chồng tôi.",
             "Tôi có thể cho bạn biết loại căng thẳng mà vợ / chồng tôi đang đối mặt trong cuộc sống của cô ấy / anh ấy.",
             "Tôi có kiến ​​thức về thế giới bên trong của vợ / chồng tôi.",
             "Tôi biết những lo lắng cơ bản của vợ / chồng tôi.",
             "Tôi biết nguồn căng thẳng hiện tại của vợ / chồng tôi là gì.",
             "Tôi biết những hy vọng và mong muốn của vợ / chồng tôi.",
             "Tôi biết người phối ngẫu của mình rất rõ.",
             "Tôi biết bạn bè của vợ / chồng tôi và các mối quan hệ xã hội của họ.",
             "Tôi cảm thấy hung hăng khi tranh cãi với vợ / chồng của mình.",
             "Khi thảo luận với vợ / chồng của tôi, tôi thường sử dụng các cụm từ như  'bạn luôn luôn' hoặc 'bạn không bao giờ'. ",
             "Tôi có thể sử dụng những câu nói tiêu cực về tính cách của vợ / chồng tôi trong các cuộc thảo luận của chúng ta.",
             "Tôi có thể sử dụng các biểu hiện xúc phạm trong các cuộc thảo luận của chúng ta.",
             "Tôi có thể xúc phạm người phối ngẫu của mình trong các cuộc thảo luận của chúng ta.",
             "Tôi có thể thấy bẽ mặt khi chúng ta thảo luận.",
             "Cuộc thảo luận của tôi với người phối ngẫu của tôi không bình tĩnh.",
             "Tôi ghét cách mở chủ đề của vợ / chồng tôi.",
             "Các cuộc thảo luận của chúng tôi thường xảy ra đột ngột.",
             "Chúng tôi chỉ bắt đầu thảo luận trước khi tôi biết chuyện gì đang xảy ra",
             "Khi tôi nói chuyện với người bạn đời của mình về điều gì đó, sự bình tĩnh của tôi đột nhiên bị phá vỡ.",
             "Khi tôi tranh cãi với vợ / chồng của mình, tôi chỉ đi ra ngoài và không nói một lời nào.",
             "Tôi chủ yếu im lặng để làm dịu môi trường một chút.",
             "Đôi khi tôi nghĩ rằng việc rời khỏi nhà một thời gian là điều tốt cho tôi.",
             "Tôi thà im lặng còn hơn thảo luận với vợ / chồng của mình.",
             "Ngay cả khi tôi đúng trong cuộc thảo luận, tôi vẫn im lặng để làm tổn thương người phối ngẫu của mình.",
             "Khi bàn bạc với bạn đời, tôi im lặng vì sợ không kiềm chế được cơn tức giận",
             "Tôi cảm thấy đúng trong các cuộc thảo luận của chúng ta.",
             "Tôi không liên quan gì đến những gì tôi đã bị buộc tội.",
             "Tôi thực sự không phải là người có tội về những gì tôi bị buộc tội.",
             "Tôi không phải là người sai về các vấn đề ở nhà.",
             "Tôi sẽ không ngần ngại nói với người bạn đời của mình về sự kém cỏi của cô ấy / anh ấy.",
             "Khi tôi thảo luận, tôi nhắc nhở người phối ngẫu của mình về sự kém cỏi của cô ấy / anh ấy",
             "Tôi không ngại nói với vợ / chồng mình về sự kém cỏi của cô ấy / anh ấy."]


#gọi file index.html và hiển thị các câu hỏi lên màn hình
@app.route('/')
def home():
    return render_template('index.html', len=len(questions), questions=questions)


#nhận yêu cầu và dự đoán kết quả
@app.route('/predict',methods=['POST'])
def predict():
    #nhận các giá trị đã chọn ở trang index
    int_features = [int(x) for x in request.form.values()]
    
    #chuyển các giá trị nhận được thành một mảng
    final_features = [np.array(int_features)]
    
    #dự đoán mảng giá trị
    prediction = model.predict(final_features)

    #trả về kết quả dự đoán 0 hoặc 1
    output = round(prediction[0], 2)

    #trả về kết quả dự đoán dạng chuỗi
    result = ""
    if(output==1):
        result = "Ly hôn"
    if(output==0):
        result = "Không ly hôn"

    #trả về kết quả ở trang results
    return render_template('results.html', prediction_text='{}'.format(result))


#hiển thị trang chọn file
@app.route('/choose_file', methods=['GET'])
def choose_file():  
    return render_template('file.html');

#nhận yêu cầu gửi file và trả về kết quả dự đoán
@app.route('/transform',methods=['POST'])
def tranform_view():
    f = request.files['data_file']

    #dùng hàm read_csv để đọc file vừa nhận được
    data = pd.read_csv(f, names=None, index_col=None)

    #lấy những cột và dòng giá trị để dự đoán
    dt = data.iloc[:, :]
    
    #dự đoán kết quả
    dubao = model.predict(dt)

    #thêm cột label và giá trị đã dự đoán được vào file 
    dt["Label"] = dubao

    #xuất ra file results.csv
    dt.to_csv('./static/outputs/results.csv', header=True)

    #mở lại file csv với tên results.csv
    file = open("./static/outputs/results.csv", "r")

    #trả về file results.csv cho người dùng
    return Response(file, mimetype="text/csv",headers={"Content-disposition":"attachment; filename=results.csv"})

#hàm main để chạy app
if __name__ == "__main__":
    app.run(debug=True)

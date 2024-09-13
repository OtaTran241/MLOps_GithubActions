# Sử dụng Python 3.9
FROM python:3.9-slim

# Đặt biến môi trường để Python không ghi đệm output
ENV PYTHONUNBUFFERED=1

# Tạo thư mục làm việc trong container
WORKDIR /app

# Copy các file cần thiết vào container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Lệnh mặc định sẽ thực hiện khi container khởi động (ví dụ, chạy training)
CMD ["python", "src/train.py"]

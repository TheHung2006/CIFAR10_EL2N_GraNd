# 1.Cơ sở lý thuyết

## 1.1. Tập dữ liệu
Giả sử tập dữ liệu huấn luyện được biểu diễn bởi:
$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}
$$

Trong đó:
- $N$: số lượng mẫu dữ liệu.
- $x_i$: đầu vào của mẫu thứ $i$.
- $y_i$: nhãn thực tế của mẫu thứ $i$.
- $C$: số lượng lớp trong bài toán phân loại.
- $\theta$: tập tham số của mô hình.
- $f_\theta(x_i)$: đầu ra của mô hình (logit) khi đưa mẫu $x_i$ vào.

Đối với bài toán phân loại, đầu ra của mô hình chuyển thành vector xác suất thông qua hàm Softmax:
$$
p_\theta(x_i)=\operatorname{softmax}(f_\theta(x_i))
$$

Với:
$$
p_\theta(x_i)=
\begin{bmatrix}
p_\theta(y=1\mid x_i)\\
p_\theta(y=2\mid x_i)\\
\vdots\\
p_\theta(y=C\mid x_i)
\end{bmatrix}
$$

Vector này biểu diễn xác suất mà mô hình dự đoán mẫu $x_i$ thuộc từng lớp.

## 1.2. Vector nhãn thực tế
Trong bài toán phân loại, nhãn $y_i$ có thể được biểu diễn dưới dạng one-hot vector:
$$
y_i \in \mathbb{R}^{C}
$$

Ví dụ, nếu có 4 lớp và mẫu thuộc lớp 2 thì:
$$
y_i=[0,1,0,0]^T
$$

# 2.Phân tích phương pháp GraNd (Gradient Normed Score)

---

## 2.1. Khái niệm

**GraNd (Gradient Normed)** là phương pháp định lượng tầm quan trọng của từng mẫu dữ liệu huấn luyện $(x_i, y_i)$ dựa trên **độ lớn (chuẩn $L_2$) của vector gradient hàm mất mát** đối với toàn bộ các tham số của mô hình.

- **Mục tiêu:** Xác định xem mẫu dữ liệu nào đóng góp nhiều nhất vào việc cập nhật trọng số và giảm hàm mất mát của mô hình ngay từ giai đoạn đầu huấn luyện (*early in training*) hoặc thậm chí ngay tại thời điểm khởi tạo ngẫu nhiên ($t=0$).
- **Nguyên lý :** Mẫu nào tạo ra vector gradient có độ dài lớn thì mẫu đó đang tạo ra lực kéo mạnh nhất để dịch chuyển trọng số của mô hình trong không gian tham số. Ngược lại, mẫu có gradient gần bằng 0 là mẫu "dư thừa", việc loại bỏ nó hầu như không ảnh hưởng đến quỹ đạo tối ưu.

---

## 2.2. Công thức toán học

Xét bài toán phân loại với mô hình có tham số $\theta_t \in \mathbb{R}^D$ tại thời điểm $t$, hàm mất mát Cross-Entropy $\ell(p_{\theta_t}(x_i), y_i)$.


### a. Định nghĩa điểm GraNd lý thuyết
Do tham số $\theta_t$ bị ảnh hưởng bởi tính ngẫu nhiên của hạt giống khởi tạo (random seed) và thứ tự gom nhóm mini-batch, **GraNd score** của mẫu $(x_i, y_i)$ được định nghĩa là kỳ vọng toán học:
$$
\chi_t(x_i, y_i) = \mathbb{E}_{\theta_t} \left[ \|g_t(x_i, y_i)\|_2 \right]
$$

### c. Công thức tính toán thực nghiệm
Trong thực tế, kỳ vọng được tính xấp xỉ bằng cách lấy trung bình cộng chuẩn gradient qua $M$ mạng nơ-ron được khởi tạo độc lập :
$$
\chi_t(x_i, y_i) \approx \frac{1}{M} \sum_{m=1}^M \|\nabla_{\theta_t^{(m)}} \ell(p_{\theta_t^{(m)}}(x_i), y_i)\|_2
$$

---

## 2.3. Cơ sở toán học 

Cơ sở lý thuyết của GraNd giải quyết câu hỏi: **"Nếu bỏ mẫu $(x_j, y_j)$ ra khỏi tập huấn luyện, tốc độ học trên một mẫu kiểm tra bất kỳ $(x^*, y^*)$ bị ảnh hưởng tối đa là bao nhiêu?"**




### Ý nghĩa toán học cốt lõi:
1. **Tính độc lập của hằng số $\kappa$:** Hệ số $\kappa$ hoàn toàn **không phụ thuộc** vào mẫu huấn luyện $(x_j, y_j)$, nó chỉ phụ thuộc vào điểm kiểm tra $(x^*, y^*)$ và tham số hiện tại của mạng.
2. **Cận trên phổ quát:** Chuẩn gradient $\|g_t(x_j, y_j)\|_2$ là đại lượng duy nhất khống chế trần mức độ ảnh hưởng của mẫu $j$ lên toàn bộ không gian dữ liệu.
   - Nếu $\|g_t(x_j, y_j)\|_2 \to 0$, thì dù mẫu kiểm tra $(x^*, y^*)$ là bất kỳ mẫu nào, độ chênh lệch cũng triệt tiêu về $0 \implies$ Mẫu $j$ không đóng góp vào việc học.
3. **Vai trò của kỳ vọng $\mathbb{E}_{\theta_t}$:** Nếu chỉ tính gradient trên một mạng duy nhất tại $t=0$, giá trị gradient bị chi phối nặng bởi sự ngẫu nhiên của trọng số. Việc lấy kỳ vọng (trung bình qua nhiều mạng) giúp triệt tiêu nhiễu ngẫu nhiên, giữ lại tín hiệu thực sự về **độ khó nội tại** của mẫu đối với bài toán.

---

## 2.4. Ví dụ minh họa

Để hình dung trực quan cách GraNd tính toán và xếp hạng dữ liệu, xét một mô hình hồi quy logistic đơn giản với 1 trọng số $w$ và không có bias: $f(x) = w \cdot x$.

- Hàm kích hoạt Sigmoid: $p(x) = \sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}}$.
- Hàm mất mát Binary Cross-Entropy: $\ell(p, y) = - [y \ln(p) + (1-y) \ln(1-p)]$.
- Gradient theo trọng số $w$:
  $$
  g(x, y) = \frac{\partial \ell}{\partial w} = (p(x) - y) \cdot x
  $$
- Chuẩn gradient:
  $$
  \|g(x, y)\|_2 = |p(x) - y| \cdot |x|
  $$

| Mẫu dữ liệu | Đầu vào| Nhãn thực tế | Logit| Xác suất dự đoán| Sai số | Gradient Norm |
|--------------|---------------:|--------------------:|-------------:|-----------------------:|------------------:|-------------:|
| Mẫu 1  | 4.0 | 1 | 2.0 | 0.8808 | 0.1192 | **0.4768** |
| Mẫu 2  | 2.0 | 0 | 1.0 | 0.7311 | 0.7311 | **1.4621** |
| Mẫu 3  | 6.0 | 1 | 3.0 | 0.9526 | 0.0474 | **0.2846** |

### Đánh giá và quyết định cắt tỉa (Data Pruning):
1. **Mẫu 2 ($\text{GraNd} = 1.46$ - Cao nhất):** Mô hình dự đoán nhầm lớp với sai số lớn ($0.73$). Vector gradient tạo ra lực kéo mạnh nhất để điều chỉnh trọng số. Đây là mẫu nằm gần ranh giới quyết định $\rightarrow$ **Bắt buộc giữ lại**.
2. **Mẫu 1 ($\text{GraNd} = 0.48$ - Trung bình):** Mô hình đã dự đoán khá tốt, đóng góp lực kéo vừa phải.
3. **Mẫu 3 ($\text{GraNd} = 0.30$ - Thấp nhất):** Mô hình đã học rất tốt mẫu này (xác suất $95\%$), gradient tiệm cận 0. Dù có bỏ mẫu này khỏi tập huấn luyện thì tốc độ học của mô hình trên các mẫu khác cũng chỉ thay đổi tối đa một lượng rất nhỏ ($\le \kappa \times 0.30$) $\rightarrow$ **Có thể loại bỏ an toàn để giảm tải dữ liệu**.

# 3. Phân tích phương pháp EL2N (Error L2-Norm Score)


## 3.1. Khái niệm

**EL2N (Error $L_2$-Norm)** là phương pháp định lượng tầm quan trọng của mẫu dữ liệu thông qua **chuẩn Euclidean ($L_2$) của vector sai số dự đoán** giữa phân phối xác suất đầu ra của mô hình và nhãn thực tế (*one-hot vector*).

- **Mục tiêu:** Kế thừa khả năng phân loại mẫu chính xác của phương pháp GraNd nhưng **loại bỏ hoàn toàn chi phí tính gradient** qua lan truyền ngược (backpropagation), giúp quá trình chấm điểm tập dữ liệu nhanh hơn gấp hàng chục đến hàng trăm lần.
- **Thời điểm áp dụng:** Được tính toán sau một vài epoch đầu tiên của quá trình huấn luyện ($t \approx 5 - 20$ epoch). Tại thời điểm này, mạng nơ-ron đã bắt đầu nắm bắt được các đặc trưng thô của bài toán và các ranh giới lớp bắt đầu hình thành ổn định.
- **Bản chất :** Một mẫu làm cho mạng nơ-ron liên tục đưa ra dự đoán lệch xa so với nhãn thực tế (vector sai số có độ dài lớn) là mẫu mang nhiều thông tin mà mạng chưa học được, do đó cần được ưu tiên giữ lại để huấn luyện.

---

## 3.2. Công thức toán học

Xét bài toán phân loại với $C$ lớp, mô hình có tham số $\theta_t$ tại epoch thứ $t$.

### a. Vector sai số dự đoán 
- Đầu ra xác suất dự đoán của mô hình qua hàm Softmax:
  $$
  p_{\theta_t}(x_i) = \operatorname{softmax}(f_{\theta_t}(x_i)) = \begin{bmatrix} p_{\theta_t}(y=1 \mid x_i) \\ p_{\theta_t}(y=2 \mid x_i) \\ \vdots \\ p_{\theta_t}(y=C \mid x_i) \end{bmatrix} \in [0, 1]^C
  $$
- Nhãn thực tế dưới dạng one-hot vector:
  $$
  y_i = \begin{bmatrix} y_i^{(1)} \\ y_i^{(2)} \\ \vdots \\ y_i^{(C)} \end{bmatrix} \in \{0, 1\}^C
  $$
- Vector sai số dự đoán của mẫu thứ $i$ là hiệu giữa xác suất dự đoán và nhãn thực tế:
  $$
  e_t(x_i) = p_{\theta_t}(x_i) - y_i \in [-1, 1]^C
  $$

### b. Chuẩn Euclidean của vector sai số
Chuẩn $L_2$ của vector sai số này được tính bằng:
$$
\|p_{\theta_t}(x_i) - y_i\|_2 = \sqrt{\sum_{c=1}^C \left( p_{\theta_t}(y=c \mid x_i) - y_i^{(c)} \right)^2}
$$

### c. Định nghĩa điểm EL2N 
Để loại bỏ tính phụ thuộc vào quỹ đạo ngẫu nhiên của một lần chạy đơn lẻ, **EL2N score** được định nghĩa là kỳ vọng trên phân phối trọng số tại thời điểm $t$:
$$
\text{EL2N}_t(x_i, y_i) = \mathbb{E}_{\theta_t} \left[ \|p_{\theta_t}(x_i) - y_i\|_2 \right]
$$

### d. Công thức tính toán thực nghiệm
Trong thực tế, điểm EL2N được tính bằng cách lấy trung bình cộng chuẩn sai số qua $M$ mạng độc lập  sau khi đã train một số ít epoch:
$$
\text{EL2N}_t(x_i, y_i) \approx \frac{1}{M} \sum_{m=1}^M \left\| p_{\theta_t^{(m)}}(x_i) - y_i \right\|_2
$$

---

## 3. Cơ sở toán học (Trọng tâm)

Cơ sở lý thuyết của EL2N là **chứng minh giải tích rằng chuẩn sai số dự đoán $\|p - y\|_2$ tỉ lệ thuận với chuẩn gradient $\|g\|_2$ của GraNd**.

### Bước 1: Khai triển Gradient theo các Logit
Hàm mất mát Cross-Entropy cho mẫu $(x_i, y_i)$ là:
$$
\ell(p_{\theta_t}(x_i), y_i) = -\sum_{c=1}^C y_i^{(c)} \ln p_{\theta_t}^{(c)}(x_i)
$$

Theo quy tắc chuỗi nhiều biến, gradient toàn phần của hàm loss đối với vector trọng số $\theta_t \in \mathbb{R}^D$ được biểu diễn qua đạo hàm theo từng logit $f_{\theta_t}^{(c)}(x_i)$:
$$
g_t(x_i, y_i) = \nabla_{\theta_t} \ell(p_{\theta_t}(x_i), y_i) = \sum_{c=1}^C \frac{\partial \ell}{\partial f_{\theta_t}^{(c)}(x_i)} \cdot \nabla_{\theta_t} f_{\theta_t}^{(c)}(x_i)
$$

Đối với sự kết hợp kinh điển giữa Softmax và Cross-Entropy, đạo hàm riêng theo logit có dạng đại số rất đẹp:
$$
\frac{\partial \ell}{\partial f_{\theta_t}^{(c)}(x_i)} = p_{\theta_t}(y=c \mid x_i) - y_i^{(c)}
$$

Đặt Jacobian của logit thứ $c$ là $J_{\theta_t}^{(c)}(x_i) = \nabla_{\theta_t} f_{\theta_t}^{(c)}(x_i) \in \mathbb{R}^D$. Khi đó:
$$
g_t(x_i, y_i) = \sum_{c=1}^C \left( p_{\theta_t}^{(c)}(x_i) - y_i^{(c)} \right) J_{\theta_t}^{(c)}(x_i)
$$

### Bước 2: Khai triển chuẩn bậc hai của Gradient
Lấy tích vô hướng của $g_t(x_i, y_i)$ với chính nó:
$$
\|g_t(x_i, y_i)\|_2^2 = \sum_{c=1}^C \sum_{k=1}^C \left( p_{\theta_t}^{(c)}(x_i) - y_i^{(c)} \right) \left( p_{\theta_t}^{(k)}(x_i) - y_i^{(k)} \right) \left\langle J_{\theta_t}^{(c)}(x_i), J_{\theta_t}^{(k)}(x_i) \right\rangle
$$

Biểu thức này cho thấy chuẩn gradient phụ thuộc vào hình học của các vector gradient logit $J_{\theta_t}^{(c)}(x_i)$ trong không gian tham số.

### Bước 3: Áp dụng các đặc tính hình học thực nghiệm (Empirical Approximations)
Dựa trên các nghiên cứu về hình học mặt phẳng hàm mất mát của mạng nơ-ron sâu (Fort & Ganguli, 2019; Fort et al., 2020), chỉ sau một vài epoch huấn luyện đầu tiên, không gian tham số xuất hiện 2 tính chất ổn định:

1. **Tính trực giao giữa các lớp (Class Orthogonality):**
   Các vector gradient logit tương ứng với các lớp khác nhau gần như trực giao đôi một trong không gian tham số hàng triệu chiều:
   $$
   \left\langle J_{\theta_t}^{(c)}(x_i), J_{\theta_t}^{(k)}(x_i) \right\rangle \approx 0 \quad (\forall c \neq k)
   $$
   *(Điều này làm triệt tiêu toàn bộ các số hạng chéo $c \neq k$ trong tổng kép).*

2. **Tính đồng nhất độ dài (Uniform Norm):**
   Độ dài của các vector gradient logit xấp xỉ đồng đều giữa các lớp và không biến động quá lớn giữa các mẫu dữ liệu:
   $$
   \|J_{\theta_t}^{(c)}(x_i)\|_2^2 \approx \lambda_t^2 \quad (\forall c \in \{1, \dots, C\})
   $$

### Bước 4: Thiết lập hệ thức tương đương
Thay hai điều kiện xấp xỉ trên vào phương trình chuẩn bậc hai:
$$
\|g_t(x_i, y_i)\|_2^2 \approx \sum_{c=1}^C \left( p_{\theta_t}^{(c)}(x_i) - y_i^{(c)} \right)^2 \cdot \|J_{\theta_t}^{(c)}(x_i)\|_2^2 \approx \lambda_t^2 \sum_{c=1}^C \left( p_{\theta_t}^{(c)}(x_i) - y_i^{(c)} \right)^2
$$

Lấy căn bậc hai hai vế:
$$
\|g_t(x_i, y_i)\|_2 \approx \lambda_t \cdot \|p_{\theta_t}(x_i) - y_i\|_2
$$

### Ý nghĩa toán học cốt lõi:
1. **Bảo toàn thứ tự xếp hạng (Ranking Preserved):** Vì hệ số $\lambda_t > 0$ là hằng số tỉ lệ chung cho cả mô hình tại epoch $t$, thứ tự quan trọng của các mẫu khi xếp theo GraNd hoàn toàn đồng dạng với thứ tự khi xếp theo EL2N:
   $$
   \|g_t(x_a, y_a)\|_2 > \|g_t(x_b, y_b)\|_2 \iff \|p_{\theta_t}(x_a) - y_a\|_2 > \|p_{\theta_t}(x_b) - y_b\|_2
   $$
2. **Đột phá về chi phí tính toán:**
   - GraNd yêu cầu tính toán trong không gian tham số $\mathbb{R}^D$ ($D \sim 10^7 - 10^8$ chiều), đòi hỏi lan truyền ngược tốn rất nhiều bộ nhớ GPU.
   - EL2N chuyển toàn bộ bài toán về không gian nhãn $\mathbb{R}^C$ ($C$ rất nhỏ, ví dụ $C=10$ hoặc $100$), chỉ cần chạy **Forward Pass**, giảm độ phức tạp tính toán từ $\mathcal{O}(N \cdot D)$ xuống $\mathcal{O}(N \cdot C)$.

---

## 3.3. Ví dụ minh họa

Xét một bài toán phân loại hình ảnh gồm 3 lớp con vật ($C = 3$):
- Lớp 1: **Mèo** $\rightarrow y = [1, 0, 0]^T$
- Lớp 2: **Chó** $\rightarrow y = [0, 1, 0]^T$
- Lớp 3: **Chim** $\rightarrow y = [0, 0, 1]^T$

Giả sử sau 10 epoch huấn luyện đầu tiên, ta đưa 3 mẫu đại diện vào mô hình và thu được vector xác suất Softmax như sau:

---

### Mẫu 1: Ảnh con Mèo rất rõ ràng (Mẫu dễ / Điển hình)
- Nhãn thực tế: $y_1 = [1, 0, 0]^T$
- Mô hình dự đoán: $p_{\theta}(x_1) = [0.92, 0.05, 0.03]^T$
- Vector sai số:
  $$
  e_1 = p_{\theta}(x_1) - y_1 = \begin{bmatrix} 0.92 - 1 \\ 0.05 - 0 \\ 0.03 - 0 \end{bmatrix} = \begin{bmatrix} -0.08 \\ 0.05 \\ 0.03 \end{bmatrix}
  $$
- Điểm EL2N:
  $$
  \text{EL2N}(x_1) = \sqrt{(-0.08)^2 + 0.05^2 + 0.03^2} = \sqrt{0.0064 + 0.0025 + 0.0009} = \sqrt{0.0098} \approx \mathbf{0.099}
  $$

---

### Mẫu 2: Ảnh Mèo chụp góc khuất / lai Chó (Mẫu khó / Nằm ở ranh giới quyết định)
- Nhãn thực tế: $y_2 = [1, 0, 0]^T$
- Mô hình phân vân giữa Mèo và Chó: $p_{\theta}(x_2) = [0.45, 0.45, 0.10]^T$
- Vector sai số:
  $$
  e_2 = p_{\theta}(x_2) - y_2 = \begin{bmatrix} 0.45 - 1 \\ 0.45 - 0 \\ 0.10 - 0 \end{bmatrix} = \begin{bmatrix} -0.55 \\ 0.45 \\ 0.10 \end{bmatrix}
  $$
- Điểm EL2N:
  $$
  \text{EL2N}(x_2) = \sqrt{(-0.55)^2 + 0.45^2 + 0.10^2} = \sqrt{0.3025 + 0.2025 + 0.0100} = \sqrt{0.5150} \approx \mathbf{0.718}
  $$

---

### Mẫu 3: Ảnh Chó nhưng bị gán nhãn nhầm là Mèo (Nhiễu nhãn / Outlier cực đoan)
- Nhãn bị gán nhầm: $y_3 = [1, 0, 0]^T$ (nhưng ảnh thực chất là Chó)
- Mô hình nhận diện rõ các đặc trưng của Chó nên dự đoán: $p_{\theta}(x_3) = [0.02, 0.96, 0.02]^T$
- Vector sai số:
  $$
  e_3 = p_{\theta}(x_3) - y_3 = \begin{bmatrix} 0.02 - 1 \\ 0.96 - 0 \\ 0.02 - 0 \end{bmatrix} = \begin{bmatrix} -0.98 \\ 0.96 \\ 0.02 \end{bmatrix}
  $$
- Điểm EL2N:
  $$
  \text{EL2N}(x_3) = \sqrt{(-0.98)^2 + 0.96^2 + 0.02^2} = \sqrt{0.9604 + 0.9216 + 0.0004} = \sqrt{1.8824} \approx \mathbf{1.372}
  $$
  

---

### Phân tích chiến lược cắt tỉa dữ liệu từ kết quả EL2N:

| Mẫu | Điểm EL2N | Nhóm dữ liệu | Xử lý |
| :--- | :---: | :--- | :--- |
| **Mẫu 1** | $\approx \mathbf{0.099}$ | **Mẫu dễ / Dư thừa** | Mô hình đã học vững chắc, gradient tác động cực nhỏ. Có thể **loại bỏ (prune)** với số lượng lớn để giảm dung lượng dữ liệu huấn luyện mà không làm giảm accuracy. |
| **Mẫu 2** | $\approx \mathbf{0.718}$ | **Mẫu cốt lõi / Ranh giới** | Nằm sát ranh giới quyết định giữa các lớp. Đây là những mẫu đóng vai trò định hình biểu diễn đặc trưng của mạng $\rightarrow$ **Bắt buộc giữ lại**. |
| **Mẫu 3** | $\approx \mathbf{1.372}$ | **Mẫu nhiễu / Outlier** | Điểm tiệm cận cực đại $\sqrt{2}$. Mẫu này thường do gán nhãn sai. Nếu ép mô hình tiếp tục học mẫu này, mạng sẽ bị méo ranh giới phân loại. Dùng **cửa sổ trượt (sliding window)** để **loại bỏ** nhóm điểm cực đại này sẽ giúp tăng test accuracy. |
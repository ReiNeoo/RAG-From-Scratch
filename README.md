Fatura Bilgi Çıkartma Uygulaması

Fatura Bilgi Çıkartma Uygulaması, farklı formatlardaki fatura dosyalarından (PDF, görüntü, XML) önemli bilgileri otomatik olarak çıkaran ve yapılandırılmış bir formatta sunan bir araçtır.


📋 Özellikler
Çoklu Format Desteği: PDF, resim (PNG, JPG), ve XML dosyalarını işleyebilme
Otomatik Bilgi Çıkarma: Toplam tutar, fatura numarası ve vergi numarası gibi önemli bilgileri otomatik olarak çıkarma
Yapay Zeka Tabanlı: YOLO nesne tespit modeli ve büyük dil modelleri (Llama3.1) kullanan gelişmiş analiz
Kullanıcı Dostu Arayüz: Gradio ile oluşturulmuş basit ve etkili web arayüzü
Doğrudan JSON Çıktısı: Yapılandırılmış, kullanılabilir veri formatı

🔧 Kurulum
Ön Koşullar

Python 3.8 veya üzeri
CUDA uyumlu bir GPU (OCR ve görüntü işleme için önerilir)
En az 8GB RAM

Adımlar

Depoyu klonlayın:
bashgit clone https://github.com/kullanici/fatura-bilgi-cikarma.git
cd fatura-bilgi-cikarma

Bağımlılıkları yükleyin:
bashpip install -r requirements.txt

Ollama'yı kurun ve Llama3.1 modelini indirin:
bash# Ollama kurulumu için https://ollama.ai adresini ziyaret edin
ollama pull Llama3.1

Gerekli modelleri indirin:
bash# YOLOv10 belge düzeni modelini indirin
mkdir -p models
# Modeli https://example.com/models/doclayout_yolo_docstructbench_imgsz1024.pt adresinden indirin 
# ve models/ dizinine yerleştirin


🚀 Kullanım

Uygulamayı başlatın:
bashpython app.py

Tarayıcınızda açılan Gradio arayüzünü kullanın veya konsolda gösterilen bağlantıyı takip edin.
"Dosya Yükleyin" alanına bir fatura dosyası sürükleyin veya tıklayarak seçin.
"Dosyayı İşle" düğmesine tıklayın (veya dosya otomatik olarak işlenmeye başlayacaktır).
Sonuçlar JSON formatında "Sonuç" alanında gösterilecektir.

📊 Örnek Çıktı
json{
  "totalAmount": "214.50 TL",
  "invoiceNumber": "FTR2023000123456",
  "sellerRegistrationNumber": "1234567890"
}
🏗️ Proje Yapısı
proje/
|-- app.py                          # Ana uygulama ve Gradio arayüzü
|-- requirements.txt                # Bağımlılıklar
|-- README.md                       # Bu dosya
|-- src/
    |-- core/
    |   |-- base.py                 # Temel soyut sınıf tanımlamaları
    |
    |-- extractors/
    |   |-- image_extractor.py      # Görüntü dosyaları için çıkarıcı
    |   |-- pdf_extractor.py        # PDF dosyaları için çıkarıcı
    |   |-- text_extractor.py       # Metin tabanlı içerik için çıkarıcı
    |   |-- xml_extractor.py        # XML dosyaları için çıkarıcı
    |
    |-- utils/
        |-- model_loader.py         # Model yükleme yardımcı sınıfları
📝 Gereksinimler
requirements.txt dosyası aşağıdaki bağımlılıkları içerir:
gradio>=4.0.0
pymupdf>=1.20.0
pillow>=8.0.0
ollama>=0.1.0
transformers>=4.30.0
torch>=2.0.0
doclayout-yolo>=1.0.0
🔍 Bilinen Sorunlar ve Kısıtlamalar

Uygulama şu anda öncelikle Türkçe faturalar için optimize edilmiştir.
Düşük kaliteli görüntüler veya standart olmayan düzenlere sahip faturalar işlenirken hatalar oluşabilir.
GPU olmayan ortamlarda görüntü işleme performansı düşük olabilir.

🛠️ Sorun Giderme

GPU Hatası: CUDA out of memory hatası alırsanız, daha küçük görüntüler kullanmayı deneyin veya imgsz parametresini düşürün.
Model Yükleme Hatası: Model dosyalarının doğru konumda olduğunu kontrol edin.
Ollama Hatası: Ollama servisinin çalıştığını ve Llama3.1 modelinin doğru şekilde yüklendiğini doğrulayın.

🤝 Katkıda Bulunma

Bu repo'yu fork edin
Yeni bir dal (branch) oluşturun (git checkout -b ozellik/muhteşem-özellik)
Değişikliklerinizi commit edin (git commit -m 'Muhteşem özellik eklendi')
Dalınızı push edin (git push origin ozellik/muhteşem-özellik)
Bir Pull Request oluşturun
/*
Template according to: https://se.moevm.info/doku.php/courses:reportrules

Latex reference from one cool guy:
https://github.com/JAkutenshi/eltechLaTeXTemplates/blob/master/LabReports/tex/title.tex
*/


// Page setup
#set page(
  width: 210mm,
  height: 297mm,
  margin: (top: 20mm, bottom: 20mm, left: 30mm, right: 15mm)
)

// General text setup
#set text(
  size: 14pt,
  lang: "ru"
)

// Paragraph setup
#set par(
  leading: 1.5em,
  first-line-indent: 1.25cm,
  justify: true
)

// To provide numeration like 1, 1.1, 1.1.1 and so on
#set enum(full: true)

// Setup level 1 header
#show heading.where(level: 1): it => [
  #set text(size: 14pt, weight: "bold")
  #set par(first-line-indent: 0pt, leading: 1.5em)
  #set align(center)
  #upper(it.body)
]

// Setup level 2 header
#show heading.where(level: 2): it => [
  #set text(size: 14pt, weight: "bold")
  #set par(first-line-indent: 1.25cm, leading: 1.5em, justify: true)
  #it.body
]

// Setup table captions
#show figure.where(kind: table): fig => {
  align(left)[
    #fig.caption
    #fig.body
  ]
}

// Long "-" between numering and caption in all figures
#show figure: set figure.caption(separator: [ --- ])

// Force all raw blocks to have 1em indent between lines
#show raw.where(block: true): set par(leading: 1em)
// Force all raw blocks to have left alignment
#show raw.where(block: true): set align(left)

// Enable formula numbering
#set math.equation(numbering: "(1)")

// First page setup
#align(center)[
  #set text(weight: "semibold")

  #set par(leading: 1em)

  МИНОБРНАУКИ РОССИИ \
  САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ \
  ЭЛЕКТРОТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ \
  «ЛЭТИ» ИМ. В.И. УЛЬЯНОВА (ЛЕНИНА) \
  Кафедра МО ЭВМ

  #v(54mm)

  ОТЧЕТ \
  по дополнительному заданию \
  по дисциплине "Введение в машинное обучение" \
  Тема: Сегментация изображений

  #v(54mm)

  #table(
    columns: (33%, 33%, 33%),
    inset: 10pt,
    align: horizon,
    stroke: none,
    "Студент гр. 3381",
    "",
    table.hline(start: 1 , end: 2),
    "Иванов А.А.",
    "Преподаватель",
    "",
    table.hline(start: 1 , end: 2),
    "Жангиров Т.Р."
  )

  #set align(bottom)
  Санкт-Петербург \
  #datetime.today().year()
]
#pagebreak()

// Start numbering here to skip first page numering
#set page(
  numbering: "1"
)

// To make indent before first header

\
== Задание
#underline[Вариант 9]

Написать скрипт, который позволяет загружать изображение. Проводить сегментацию на указанное количество участков (используя метод кластеризации), а затем выводит изображение с полученными участками, а также участки исходного изображения, к которым применили отдельный сегмент как маску.

\
== Теория

Сегментация изображений -- процесс разбиения изображения на однородные, осмысленные регионы. Позволяет выделять на изображении объекты и анализировать структуру изображений.

\
== Выполнение работы

В данной работе для сегментации изображения использовал метод кластеризации K-Means. Использована библиотечная реализация `sklearn.cluster.KMeans`. Изображение выводится с помощью функции `plt.imshow(image_array)`.

Исходное изображение показано на @cat.

#figure(
  image("images/cat.jpg", width: 50%),
  caption: [Исходное изображение.]
) <cat>

С помощью кода в @segm_code производится показ на экране кластеризованного изображения
#figure(
  ```python
  segmented_image = \
    kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
  plt.imshow(segmented_image / 255)
  ```,
  caption: [
    Код для сегментации изображения.
  ]
) <segm_code>

Т.е. после обучения модели KMeans на изображении берутся цвета из середины каждого кластера и применяются к всем пикселям соответствующих кластеров.

#figure(
  image("images/segmented_cat.jpg", width: 50%),
  caption: [Изображение, разделённое на 5 кластеров.]
) <segmented_cat>

Для того, чтобы вывести только пиксели, которые находятся в соответствующих кластерах использован код, представленный в @mask_code.

#figure(
  ```python
  CLUSTERS_TO_SHOW = [1]

  mask = np.isin(kmeans.labels_, CLUSTERS_TO_SHOW)
  masked_image = np.where(mask[:, None], X, 0).reshape(image.shape)
  plt.imshow(masked_image / 255)
  ```,
  caption: [Код для показа определённых кластеров.]
) <mask_code>

Результат представлен на @masked_cat.

#figure(
  image("images/masked_cat.jpg", width: 50%),
  caption: [Пиксели кластера с индексом 1.]
) <masked_cat>


#pagebreak()
= Приложение А \ ИСХОДНЫЙ КОД

Ниже представлен исходный код программы.

```python
# %% <- represents divider between cells in .ipynb, or in REPL

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans


IMAGE_PATH = "cat.jpg"
N_CLUSTERS = 5

# %%

image = plt.imread(IMAGE_PATH)
print(f"Original image size: {image.shape}")
plt.imshow(image)

X = np.reshape(image, (-1, 3))
print(f"Reshaped image size: {X.shape}")

# %%

kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(X)

# %%

segmented_image = \
  kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
plt.imshow(segmented_image / 255)

# %%

CLUSTERS_TO_SHOW = [1]

mask = np.isin(kmeans.labels_, CLUSTERS_TO_SHOW)
masked_image = np.where(mask[:, None], X, 0).reshape(image.shape)
plt.imshow(masked_image / 255)
```

 
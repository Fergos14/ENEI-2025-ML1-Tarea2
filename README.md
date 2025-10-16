## Assignment: Logistic Regression and Multiclass Extensions

**Deadline:** Monday, October 13th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ucimlrepo`.

**Integrante:** Rodrigo Caballero


---
###  How the gradient differs between binary, OvA, and multinomial forms
En la **regresión logística binaria**, el gradiente se calcula en función de la diferencia entre las probabilidades predichas por el modelo y las verdaderas etiquetas. Solo se ajusta un conjunto de pesos, lo que hace que el proceso sea directo y enfocado en una única frontera de decisión entre dos clases.

En el enfoque **One-vs-All (OvA)**, el problema se divide en varios submodelos binarios. Cada uno aprende a distinguir una clase frente a todas las demás, por lo que cada clasificador tiene su propia gradiente y conjunto de pesos. El cálculo del gradiente es el mismo que en el caso binario, pero se realiza de forma independiente para cada modelo. Esto implica mayor redundancia y más iteraciones totales, ya que cada clasificador aprende patrones similares sin compartir información con los otros.

En la **regresión logística multinomial (softmax)**, los gradientes de todas las clases se actualizan simultáneamente. Cada clase ajusta sus pesos considerando también el efecto que ese cambio tendrá sobre las demás clases, ya que las probabilidades están acopladas y deben sumar uno. Este enfoque permite aprovechar información compartida y suele converger más rápido, evitando el entrenamiento separado de múltiples modelos.

---
###  How numerical stability issues may arise in softmax
Cuando los valores que entran al **softmax** son muy altos (**logit**), las exponenciales pueden crecer demasiado y dar números enormes que la computadora no puede manejar correctamente. Esto provoca que las operaciones den resultados extraños o inválidos, como valores infinitos o errores en el cálculo. Esto afecta directamente la función de costo y puede impedir la convergencia del modelo. Para evitar este problema, se aplica una **corrección de estabilidad numérica**: antes de exponenciar los **logits**, se resta el valor máximo de cada fila. Este ajuste mantiene las relaciones relativas entre los valores y evita que las exponenciales crezcan demasiado. Aunque el resultado final del **softmax** no cambia, este paso es fundamental para evitar desbordamientos y mantener un entrenamiento estable. Sin esta corrección, las probabilidades pueden volverse inconsistentes y el descenso de gradiente incluso puede divergir.

---

###  When OvA and multinomial approaches diverge in predictions
Las diferencias entre ambos enfoques se resaltan cuando las clases no están bien separadas o se mezclan entre sí. En el modelo **One-vs-All**, cada clasificador trabaja por su cuenta, así que puede pasar que más de una clase tenga una probabilidad alta al mismo tiempo. Eso hace que las predicciones sean menos claras o incluso contradictorias. En cambio, el modelo **multinomial (softmax)** obliga a que las clases compitan entre sí, porque todas las probabilidades tienen que sumar uno. Esto hace que las decisiones sean más coherentes y fáciles de interpretar. En este caso, como el **dataset de vinos** está muy bien definido, ambos métodos dieron exactamente el mismo resultado. Pero en situaciones más reales, donde las fronteras entre clases no son tan evidentes o las variables están muy correlacionadas, el modelo **softmax** suele comportarse mejor, ya que considera la competencia entre las clases al mismo tiempo y mantiene una **distribución de probabilidades más equilibrada**.


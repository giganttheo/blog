---
title: "Intro to DPP"
date: "2020-07-01"
draft: false
---


## Finite Determinantal Point Processes :

This blog post is adapted from a school project ( in French ).
The report is available [**here**](https://github.com/giganttheo/DPP/blob/master/28_Rapport%20Tuteur%20Cassiop%C3%A9e.pdf), and the companion python notebook [**here**](https://github.com/giganttheo/DPP/blob/master/DPP_Notebook.ipynb).

Most of our work is based on Alex Kulesza and Ben Taskar's papers, mostly
_Determinantal point processes for machine learning_.
We added extra and clearer proofs.

## Definition :

Quickly :

- a point process is defined as a measure of probability,

- a point process is *determinantal* when there is a matrix $K$, known as a Kernel matrix, such that $\rm \mathbb{P}(X \in \mathcal{P(Y}|Y \subset X ) = \text{det}(K_Y)$ where $K_Y$ is the restriction of $K$ to the elements of $Y$.

- Another definition of a determinantal point process is by the $L$-ensembles : a measure of probability $\rm \mathbb{P}_L(Y = y) \propto \text{det}(L_Y)$ with $L_Y$ related to $K_Y$ by the formula :
$$K = L(L + I)^{-1}$$

The interesting thing with determinantal point processes is mainly to sample from them to have some data with a repulsive phenomenon. For that matter, the simplest algorithm is known as the *Spectral Method* and is defined as follow :

![Spectral Method algorithm](https://i.ibb.co/pKKgf8Y/Capture.png)

## Code

Now that we know a little about finite determinantal point processes, let's go in Python
and code the spectral algorithm to sample from finite DPPs defined by the eigenvectors &
eigenvalues of $K_L$ :

```python
def spectral_method(lambda_L, v_L):
  N = lambda_L.shape[0]
  J = [] # J is the set of the indices of the eigenvectors in the elementary DPP which we sample from in the second part of the algorithm
  for n in range(N):
      if random.random() < (lambda_L[n] / (1 + lambda_L[n])):
        # each eigenvector has a probability `lambda_L[n] / (1 + lambda_L[n])` of being part of J
        J.append(n)
  V = [v_L[n] for n in J]
  Y = []
  #End of the first part of the algorithm : now we will sample from the elementary DPP defined by the eigenvectors in V
  while len(V) > 0:
      rd = random.random()
      s = 0
      i = 0
      while rd > s:
        # this loop is for choosing $i$ in $\mathcal(Y)$ with a probability $\sigma_{v \in V}(np.dot(v.T,ei)**2) / len(V)$
        ei = [0 for _ in range(N) ]
        ei[i] = 1
        ei = np.array(ei)
        for v in V:
          s += (np.dot(v.T,ei)**2) / len(V) 
        i += 1
      Y.append(i)
      if len(V) > 1 :
          # We project V on the orthogonal complement of ej then deduce an orthonormal basis V_orth of rank Card(V) - 1
          V_orth = []
          for v in V :
              V_orth.append(v - np.dot(v.T,ei)*ei)
          # the new V_J is an orthogonal basis
          V = scipy.linalg.orth(np.array(V_orth[:-1]).T).T
          # we replace V by this new orthonormal basis, the cardinality of V is reduce by 1 at each loop
      else :
        # if the cardinality of V is 1, we exit the loop
        V = []
  return Y
```

Thanks for reading me, if you have any comment or questions, you can join me @ theo.gigant@telecom-sudparis.eu

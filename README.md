<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b>
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art pretrained models for inference and training</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>
<p>
Transformers act as the model-definition framework for state-of-the-art machine learning models in text, computer 
vision, audio, video, and multimodal model, for both inference and training. Visit the upstream Hugging Face for [installation instructions](https://github.com/huggingface/transformers).

## Huggingface-transformer-2025-fork-

This architecture, after initial training, distills subsequent datasets and rulesets as an extensible relational database. The inititial (stochastic) model is then refined via the integration of a theorem prover into a decidable (i.e., deterministic) architecture, which employs next-token generation via computable logic. Ontology synergy-matching capabilities are employed in situations where next-token prediction constraints are inadequate.

Builtin Production Grammar

Provided, is an **elegant proof kernel** grounded in **type theory**, inspired by the **Curry–Howard correspondence**, and minimal enough to be understandable, extensible, and verifiable. It models **intuitionistic propositional logic** using **dependent types**, and serves as a foundational "proof kernel" in the style of **pure type systems (PTS)** and **minimalist theorem provers** like those used in *Coq*, *Lean*, or *Agda*.

---

## Elegant Type-Theoretic Proof Kernel

This kernel is written in **dependently typed pseudocode**, akin to **Lean 4** or **Agda**, but adaptable to any host language supporting type theory.

### Core Logical Language

We define the logical propositions as types:

```lean
-- Type of propositions
inductive Prop : Type
| imp  : Prop → Prop → Prop       -- implication A → B
| and  : Prop → Prop → Prop       -- conjunction A ∧ B
| or   : Prop → Prop → Prop       -- disjunction A ∨ B
| not  : Prop → Prop              -- negation ¬A
| top  : Prop                     -- ⊤ (truth)
| bot  : Prop                     -- ⊥ (falsehood)
| var  : String → Prop            -- atomic propositions
```

---

## Proof Terms (Proof Objects)

Each proposition corresponds to a type. A **proof** of a proposition is a term of that type. The Curry–Howard correspondence yields the following inductive structure for proof terms:

```lean
-- Proofs-as-terms
inductive Proof : Prop → Type
| ax    : ∀ {P : Prop}, Proof P → Proof P              -- trivial wrapper (for extensionality)
| assume : ∀ {P : Prop}, Proof P                       -- assume P (used for hypotheses)
| impI  : ∀ {P Q : Prop}, (Proof P → Proof Q) → Proof (Prop.imp P Q)  -- → introduction
| impE  : ∀ {P Q : Prop}, Proof (Prop.imp P Q) → Proof P → Proof Q    -- → elimination (modus ponens)
| andI  : ∀ {P Q : Prop}, Proof P → Proof Q → Proof (Prop.and P Q)    -- ∧ introduction
| andE₁ : ∀ {P Q : Prop}, Proof (Prop.and P Q) → Proof P              -- ∧ elimination left
| andE₂ : ∀ {P Q : Prop}, Proof (Prop.and P Q) → Proof Q              -- ∧ elimination right
| orI₁  : ∀ {P Q : Prop}, Proof P → Proof (Prop.or P Q)               -- ∨ introduction left
| orI₂  : ∀ {P Q : Prop}, Proof Q → Proof (Prop.or P Q)               -- ∨ introduction right
| orE   : ∀ {P Q R : Prop}, Proof (Prop.or P Q) → (Proof P → Proof R) → (Proof Q → Proof R) → Proof R
| notI  : ∀ {P : Prop}, (Proof P → Proof Prop.bot) → Proof (Prop.not P)  -- ¬ introduction
| notE  : ∀ {P : Prop}, Proof (Prop.not P) → Proof P → Proof Prop.bot    -- ¬ elimination
| exfalso : ∀ {P : Prop}, Proof Prop.bot → Proof P                      -- ⊥ elimination (ex falso quodlibet)
| triv  : Proof Prop.top                                               -- trivial proof of ⊤
```

---

## Example: Proof of `A ∧ B → A`

This corresponds to the type-theoretic function:

```lean
def and_left {A B : Prop} : Proof (Prop.imp (Prop.and A B) A) :=
  Proof.impI (λ hAB => Proof.andE₁ hAB)
```

Which in more standard natural deduction:

```
Assume h : A ∧ B
Then from h, extract A via ∧ elimination left
Hence, A
```

---

## Example: Proof of `A → (B → A ∧ B)`

```lean
def intro_and {A B : Prop} : Proof (Prop.imp A (Prop.imp B (Prop.and A B))) :=
  Proof.impI (λ hA =>
    Proof.impI (λ hB =>
      Proof.andI hA hB))
```

---

## Example: Proof of `¬A → (A → ⊥)`

This is definitionally true by the definition of `¬A := A → ⊥`, so:

```lean
def notA_def {A : Prop} : Proof (Prop.imp (Prop.not A) (Prop.imp A Prop.bot)) :=
  Proof.impI (λ hNotA => 
    Proof.impI (λ hA => 
      Proof.notE hNotA hA))
```

---

## Advanced Example: Prove the Contrapositive

Prove `A → B ⊢ ¬B → ¬A`
(i.e., `A → B` implies `¬B → ¬A`)

```lean
def contrapositive {A B : Prop} : Proof (Prop.imp (Prop.imp A B) (Prop.imp (Prop.not B) (Prop.not A))) :=
  Proof.impI (λ hAB => 
    Proof.impI (λ hNotB =>
      Proof.notI (λ hA => 
        Proof.notE hNotB (Proof.impE hAB hA))))
```

---

## Notes on the Elegance

* This kernel leverages **inductive datatypes** over `Prop` and `Proof` to encode rules of inference.
* There is **no computation**—it’s a **pure proof checker**, not an evaluator or reducer. That would be layered atop, as in Lean or Agda.
* The kernel is **syntax-directed**; every term mirrors a proof rule.
* It aligns with **natural deduction** and **type theory** seamlessly.

---

## Extensibility

* To support **quantifiers**, add:

  * `Prop.forall : (α : Type) → (α → Prop) → Prop`
  * `Prop.exists : (α : Type) → (α → Prop) → Prop`
* Add dependent types over universes `Type u` as needed.
* For classical logic, extend with `double negation elimination`, `excluded middle`, etc.

---

## Further Reading

* *The Calculus of Constructions* by Coquand & Huet
* *Types and Programming Languages* by Benjamin C. Pierce
* *Certified Programming with Dependent Types* (CPDT) by Adam Chlipala
* *Lean 4 Theorem Proving Manual*
  ([https://leanprover.github.io/theorem\_proving\_in\_lean4/](https://leanprover.github.io/theorem_proving_in_lean4/))

**Future Prototypes**: translate this into **executable Lean 4**, or integrate **tactics** and **custom proof automation**?

</p>

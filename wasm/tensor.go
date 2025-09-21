package main

import (
	"fmt"
	"math"
)

// --- GradFn Enum --- //
type GradFn string

const (
	ADD  GradFn = "add"
	SUB  GradFn = "sub"
	MUL  GradFn = "mul"
	DIV  GradFn = "div"
	POW  GradFn = "pow"
	MEAN GradFn = "mean"
)

// --- Tensor --- //
type Tensor struct {
	Data         []float64
	RequiresGrad bool
	GradFn       *GradFn
	Grad         []float64
	Parents      []*Tensor
}

// makeTensor wraps data into a Tensor
func makeTensor(data []float64, requiresGrad bool) *Tensor {
	t := &Tensor{
		Data:         data,
		RequiresGrad: requiresGrad,
	}
	if requiresGrad {
		t.Grad = make([]float64, len(data))
	}
	return t
}

// Scalar helper
func scalar(v float64, requiresGrad bool) *Tensor {
	return makeTensor([]float64{v}, requiresGrad)
}

// matchShape (broadcast scalar to vector)
func (t *Tensor) matchShape(grad []float64) []float64 {
	// Case 1: Tensor is scalar → reduce grad to a single number
	if len(t.Data) == 1 && len(grad) > 1 {
		sum := 0.0
		for _, v := range grad {
			sum += v
		}
		return []float64{sum}
	}

	// Case 2: Broadcasting a scalar grad → expand
	if len(grad) == 1 && len(t.Data) > 1 {
		out := make([]float64, len(t.Data))
		for i := range out {
			out[i] = grad[0]
		}
		return out
	}

	// Case 3: Must match
	if len(grad) != len(t.Data) {
		panic(fmt.Sprintf("grad shape mismatch: got %d, expected %d", len(grad), len(t.Data)))
	}
	return grad
}

// --- Ops --- //

func (t *Tensor) Add(o *Tensor) *Tensor {
	data := make([]float64, max(len(t.Data), len(o.Data)))
	for i := range data {
		data[i] = t.Data[i%len(t.Data)] + o.Data[i%len(o.Data)]
	}
	fn := ADD
	return &Tensor{Data: data, RequiresGrad: true, GradFn: &fn, Parents: []*Tensor{t, o}}
}

func (t *Tensor) Sub(o *Tensor) *Tensor {
	data := make([]float64, max(len(t.Data), len(o.Data)))
	for i := range data {
		data[i] = t.Data[i%len(t.Data)] - o.Data[i%len(o.Data)]
	}
	fn := SUB
	return &Tensor{Data: data, RequiresGrad: true, GradFn: &fn, Parents: []*Tensor{t, o}}
}

func (t *Tensor) Mul(o *Tensor) *Tensor {
	data := make([]float64, max(len(t.Data), len(o.Data)))
	for i := range data {
		data[i] = t.Data[i%len(t.Data)] * o.Data[i%len(o.Data)]
	}
	fn := MUL
	return &Tensor{Data: data, RequiresGrad: true, GradFn: &fn, Parents: []*Tensor{t, o}}
}

func (t *Tensor) Pow(p *Tensor) *Tensor {
	data := make([]float64, len(t.Data))
	for i := range t.Data {
		data[i] = math.Pow(t.Data[i], p.Data[0]) // scalar exp for now
	}
	fn := POW
	return &Tensor{Data: data, RequiresGrad: true, GradFn: &fn, Parents: []*Tensor{t, p}}
}

func (t *Tensor) Mean() *Tensor {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	mean := sum / float64(len(t.Data))
	fn := MEAN
	return &Tensor{Data: []float64{mean}, RequiresGrad: true, GradFn: &fn, Parents: []*Tensor{t}}
}

// --- Autograd --- //

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	}
}

func (t *Tensor) Backward(grad ...float64) {
	g := grad
	if len(g) == 0 {
		g = []float64{1.0}
	}
	g = t.matchShape(g)

	if t.RequiresGrad {
		if t.Grad == nil {
			t.Grad = make([]float64, len(t.Data))
		}
		for i := range t.Grad {
			t.Grad[i] += g[i]
		}
	}

	if t.GradFn == nil {
		return
	}

	switch *t.GradFn {
	case ADD:
		t.Parents[0].Backward(g...)
		t.Parents[1].Backward(g...)
	case SUB:
		t.Parents[0].Backward(g...)
		neg := make([]float64, len(g))
		for i, v := range g {
			neg[i] = -v
		}
		t.Parents[1].Backward(neg...)
	case MUL:
		a, b := t.Parents[0], t.Parents[1]
		gradA := make([]float64, len(a.Data))
		gradB := make([]float64, len(b.Data))
		for i := range g {
			gradA[i%len(a.Data)] += g[i] * b.Data[i%len(b.Data)]
			gradB[i%len(b.Data)] += g[i] * a.Data[i%len(a.Data)]
		}
		a.Backward(gradA...)
		b.Backward(gradB...)
	case POW:
		base, exp := t.Parents[0], t.Parents[1]
		gradBase := make([]float64, len(base.Data))
		for i := range base.Data {
			gradBase[i] = g[0] * exp.Data[0] * math.Pow(base.Data[i], exp.Data[0]-1)
		}
		base.Backward(gradBase...)
	case MEAN:
		x := t.Parents[0]
		gradX := make([]float64, len(x.Data))
		for i := range x.Data {
			gradX[i] = g[0] / float64(len(x.Data))
		}
		x.Backward(gradX...)
	}
}

// --- Optimizer --- //

type SGD struct {
	Params []*Tensor
	LR     float64
}

func (opt *SGD) Step() {
	for _, p := range opt.Params {
		if p.RequiresGrad && p.Grad != nil {
			for i := range p.Data {
				p.Data[i] -= opt.LR * p.Grad[i]
			}
		}
	}
}

func (opt *SGD) ZeroGrad() {
	for _, p := range opt.Params {
		p.ZeroGrad()
	}
}

// --- Helpers --- //

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// --- Main: Train linear regression --- //

// func main() {
// 	// training data: y = 2x + 5
// 	xs := make([]float64, 20)
// 	ys := make([]float64, 20)
// 	for i := 0; i < 20; i++ {
// 		xs[i] = -5 + float64(i)*(10.0/19.0)
// 		ys[i] = 2*xs[i] + 5
// 	}
// 	xTensor := makeTensor(xs, false)
// 	yTensor := makeTensor(ys, false)

// 	// init params
// 	w := scalar(rand.NormFloat64(), true)
// 	b := scalar(rand.NormFloat64(), true)
// 	optimizer := SGD{Params: []*Tensor{w, b}, LR: 0.01}

// 	for epoch := 0; epoch < 200; epoch++ {
// 		optimizer.ZeroGrad()

// 		preds := w.Mul(xTensor).Add(b)
// 		diff := preds.Sub(yTensor)
// 		loss := diff.Pow(scalar(2.0, false)).Mean()

// 		loss.Backward()
// 		optimizer.Step()

// 		if epoch%20 == 0 {
// 			fmt.Printf("Epoch %d: loss=%.4f, w=%.4f, b=%.4f\n",
// 				epoch, loss.Data[0], w.Data[0], b.Data[0])
// 		}
// 	}
// }

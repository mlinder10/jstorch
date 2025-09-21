// wasm_tensor.go
// go build -o main.wasm -target js/wasm (see build command below)
package main

import (
	"math/rand"
	"syscall/js"
	"time"
)

// Assume your Tensor, makeTensor, scalar, SGD, etc. are in the same package (or import them)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Helper: convert JS Float64Array to Go []float64
func jsFloat64ArrayToSlice(v js.Value) []float64 {
	if v.IsUndefined() || v.IsNull() {
		return nil
	}
	length := v.Get("length").Int()
	out := make([]float64, length)
	// copy element by element (fast enough for moderate sizes)
	for i := 0; i < length; i++ {
		out[i] = v.Call("at", i).Float()
	}
	return out
}

// Helper: create a JS Float64Array from Go []float64
func sliceToJSFloat64Array(s []float64) js.Value {
	ta := js.Global().Get("Float64Array").New(len(s))
	for i, val := range s {
		ta.SetIndex(i, val)
	}
	return ta
}

// trainLinear(xs: Float64Array, ys: Float64Array, epochs: number, lr: number) -> { w, b, losses }
func trainLinear(this js.Value, args []js.Value) any {
	if len(args) < 4 {
		js.Global().Call("console.error", "trainLinear expects (xs, ys, epochs, lr)")
		return nil
	}
	xsVal := args[0]
	ysVal := args[1]
	epochs := args[2].Int()
	lr := args[3].Float()

	xs := jsFloat64ArrayToSlice(xsVal)
	ys := jsFloat64ArrayToSlice(ysVal)
	if len(xs) != len(ys) {
		js.Global().Call("console.error", "xs and ys must have same length")
		return nil
	}

	// Build tensors: use your Tensor helpers
	xTensor := makeTensor(xs, false)
	yTensor := makeTensor(ys, false)

	// initialize params
	w := scalar(rand.NormFloat64(), true)
	b := scalar(rand.NormFloat64(), true)
	optimizer := SGD{Params: []*Tensor{w, b}, LR: lr}

	losses := make([]float64, 0, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		optimizer.ZeroGrad()

		preds := w.Mul(xTensor).Add(b) // broadcasting handled in your code
		diff := preds.Sub(yTensor)     // elementwise
		loss := diff.Pow(scalar(2.0, false)).Mean()

		loss.Backward()
		optimizer.Step()

		losses = append(losses, loss.Data[0]) // loss is scalar Tensor with Data[0]
	}

	// return { w: w.Data[0], b: b.Data[0], losses: Float64Array }
	ret := js.Global().Get("Object").New()
	ret.Set("w", w.Data[0])
	ret.Set("b", b.Data[0])
	ret.Set("losses", sliceToJSFloat64Array(losses))
	return ret
}

func main() {
	// expose function to global scope as "trainLinear"
	js.Global().Set("trainLinear", js.FuncOf(trainLinear))

	// Prevent program from exiting
	select {}
}

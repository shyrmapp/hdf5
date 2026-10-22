package hdf5_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/shyrmapp/hdf5"
)

// Open a file with a read limit and walk every group and dataset.
func ExampleOpen() {
	f, err := hdf5.Open("testdata/with_groups.h5", hdf5.WithMaxReadBytes(256<<20))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()

	f.Walk(func(path string, obj hdf5.Object) {
		switch obj.(type) {
		case *hdf5.Group:
			fmt.Println("group  ", path)
		case *hdf5.Dataset:
			fmt.Println("dataset", path)
		}
	})
	// Output:
	// group   /
	// dataset /dataset1
	// group   /subgroup/
	// dataset /subgroup/dataset2
	// group   /subgroup/nested_group/
	// dataset /subgroup/nested_group/nested_data
}

// Read one row of a 4x5 dataset without loading the rest.
func ExampleDataset_ReadSlice() {
	f, err := hdf5.Open("testdata/with_attributes.h5")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()

	for _, obj := range f.Root().Children() {
		ds, ok := obj.(*hdf5.Dataset)
		if !ok || ds.Name() != "dataset1" {
			continue
		}
		row, err := ds.ReadSlice([]uint64{2, 0}, []uint64{1, 5})
		fmt.Println(row, err)
	}
	// Output: [10 11 12 13 14] <nil>
}

// Read attribute values; their Go type follows the stored HDF5 type.
func ExampleGroup_Attributes() {
	f, err := hdf5.Open("testdata/with_attributes.h5")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()

	attrs, err := f.Root().Attributes()
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, a := range attrs {
		v, err := a.ReadValue()
		fmt.Printf("%s = %v (%T) %v\n", a.Name, v, v, err)
	}
	// Unordered output:
	// array_attr = [1 2 3 4 5] ([]int64) <nil>
	// pi = 3.14159 (float64) <nil>
	// title = Test File (string) <nil>
	// version = 1 (int64) <nil>
}

// Write a dataset and read it back.
func ExampleCreateForWrite() {
	dir, err := os.MkdirTemp("", "hdf5-example")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "out.h5")

	fw, err := hdf5.CreateForWrite(path, hdf5.CreateTruncate)
	if err != nil {
		fmt.Println(err)
		return
	}
	ds, err := fw.CreateDataset("/temperature", hdf5.Float64, []uint64{3})
	if err == nil {
		err = ds.Write([]float64{21.5, 22.0, 22.5})
	}
	if cerr := fw.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		fmt.Println(err)
		return
	}

	f, err := hdf5.Open(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	f.Walk(func(p string, obj hdf5.Object) {
		if ds, ok := obj.(*hdf5.Dataset); ok {
			vals, err := ds.Read()
			fmt.Println(p, vals, err)
		}
	})
	// Output: /temperature [21.5 22 22.5] <nil>
}

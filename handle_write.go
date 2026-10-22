package hdf5

import "errors"

var (
	errWriterClosed   = errors.New("hdf5: file writer is closed")
	errObjectDeleted  = errors.New("hdf5: object was deleted")
	errWriterNotReady = errors.New("hdf5: invalid dataset writer")
)

// checkOpen returns an error if the writer was closed (or never opened).
func (fw *FileWriter) checkOpen() error {
	if fw == nil || fw.writer == nil {
		return errWriterClosed
	}
	return nil
}

// track remembers dw so Delete can invalidate it.
func (fw *FileWriter) track(dw *DatasetWriter) {
	if dw == nil {
		return
	}
	if fw.liveWriters == nil {
		fw.liveWriters = make(map[uint64][]*DatasetWriter)
	}
	fw.liveWriters[dw.address] = append(fw.liveWriters[dw.address], dw)
}

// markDeleted invalidates every dataset writer for the object at addr.
func (fw *FileWriter) markDeleted(addr uint64) {
	for _, dw := range fw.liveWriters[addr] {
		dw.deleted = true
	}
	delete(fw.liveWriters, addr)
}

// usable checks that the writer's file is open and its object still exists.
func (dw *DatasetWriter) usable() error {
	if dw == nil || dw.fileWriter == nil {
		return errWriterNotReady
	}
	if dw.deleted {
		return errObjectDeleted
	}
	return dw.fileWriter.checkOpen()
}

// writable additionally requires that raw data can be written to the dataset.
func (dw *DatasetWriter) writable() error {
	if err := dw.usable(); err != nil {
		return err
	}
	return dw.dataWriteErr
}

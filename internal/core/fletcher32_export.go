package core

// Fletcher32 is H5_checksum_fletcher32, for writers of the Fletcher32 filter.
func Fletcher32(data []byte) uint32 { return fletcher32(data) }

path = "/Users/preibischs/python/sofima/pass03-scale5/";

run("HDF5/N5/Zarr/OME-NGFF ... ", "url="+path+"flow_raw.zarr/");
rename("flow_raw");

run("HDF5/N5/Zarr/OME-NGFF ... ", "url="+path+"flow_cleaned.zarr/");
rename("flow_cleaned");

run("HDF5/N5/Zarr/OME-NGFF ... ", "url="+path+"inverse_map.zarr/");
rename("inverse_map");

run("HDF5/N5/Zarr/OME-NGFF ... ", "url="+path+"aligned.zarr/");
rename("aligned");

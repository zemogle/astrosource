from astropy.table import Table, Column
import numpy as np

def mock_vizier_query_region(*args, **kwargs):
    ra = np.array([154.74983, 154.90837, 154.98908, 155.03358, 155.13282])
    dec = np.array([-9.90814, -9.80628, -9.66889, -9.97419, -9.73022])
    t = Table([ra, dec], names=('RAJ2000', 'DEJ2000'), dtype=('float64', 'float64'))
    return t

def mock_vizier_query_region_vsx(*args, **kwargs):
    t = mock_vizier_query_region()
    q = {'B/vsx/vsx' : t }
    return q

def mock_vizier_query_region_apass_b(*args, **kwargs):
    t = mock_vizier_query_region()
    bmag = Column(np.array([16.82 ,    np.nan, 16.287, 16.566, 17.006], dtype=np.float32))
    ebmag = Column(np.array([0.018,   np.nan, 0.009, 0.02 , 0.11 ], dtype=np.float32))
    t.add_column(bmag, name='Bmag')
    t.add_column(ebmag, name='e_Bmag')
    q = {'II/336/apass9' : t }
    return q

def mock_vizier_query_region_apass_v(*args, **kwargs):
    t = mock_vizier_query_region()
    vmag = Column(np.array([16.82 ,    np.nan, 16.287, 16.566, 17.006], dtype=np.float32))
    evmag = Column(np.array([0.018,   np.nan, 0.009, 0.02 , 0.11 ], dtype=np.float32))
    t.add_column(vmag, name='Vmag')
    t.add_column(evmag, name='e_Vmag')
    q = {'II/336/apass9' : t }
    return q

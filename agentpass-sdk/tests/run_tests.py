import sys
sys.path.insert(0, 'src')

import pytest
pytest.main([__file__, '-v', '--tb=short'])

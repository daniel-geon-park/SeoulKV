# SeoulKV4 PoC 라이브러리 인터페이스

SeoulKV4는 한 개의 고정 flat K/V cache를 대상으로 하는 sparse decode attention
라이브러리입니다. Paged attention, cache append, 요청 스케줄링은 지원하지 않습니다.

## 1. 공개 API: 실행 객체 하나를 중심으로

| 이름 | 역할 |
|---|---|
| `SeoulKV4` | 고정 K/V 한 세트의 device workspace와 현재 네 program을 소유하고 실행합니다. |
| `SeoulKV4Config` | 검증된 실행 profile의 제한적인 설정입니다. |
| `SeoulKV4Index` | K에서 빌드하거나 기존 Stage-I NPZ에서 읽는 query-independent CPU index입니다. |
| `DecodeResult` | 재사용되는 device output의 view와 명시적 host readback을 제공합니다. |

별도의 request/session/plan 객체는 없습니다. 한 `SeoulKV4` instance는
고정 context의 한 cache snapshot에 대응합니다. 같은 device에서 다른 instance와
workspace를 동시에 공유하거나 interleave하는 기능은 아직 보장하지 않습니다.

### 사용 예시

```python
from seoulkv4 import SeoulKV4, SeoulKV4Config, SeoulKV4Index

# device는 호출자가 열어 둔 TTNN MeshDevice입니다.
# Blackhole 런타임과 L1 설정은 TT_RUNTIME.md를 참조합니다.
# k/v: CPU BF16 [4, T, 256], k는 post-RoPE, v는 원래 value입니다.
index = SeoulKV4Index.load("stage1_index.npz")
config = SeoulKV4Config()

with SeoulKV4(device=device, k=k, v=v, index=index, config=config) as attention:
    # Q는 생성자/prepare에 넣지 않고 매번 run()에 전달합니다.
    # q: CPU BF16 [1, 1, 16, 256], post-RoPE
    attention.run(q0)                # 첫 호출: lazy build + warmup, 비측정
    attention.synchronize()

    result = attention.run(q1)       # 이후: Q upload + 현재 cascade enqueue
    y = result.to_torch()            # 명시적 sync/readback, [1, 1, 16, 256]

    result = attention.run(q2)       # 과거 Q를 재사용하지 않습니다.
    y2 = result.to_torch()
```

첫 호출도 output을 반환합니다. 위 예제에서는 그 호출을 초기화용으로 사용합니다.

## 2. 클래스 계약

### `SeoulKV4Index`

```python
class SeoulKV4Index:
    @classmethod
    def build(cls, k, v=None, *, sink=128, local=256, nprobe=384,
              lam=0.0625, seed=0, build_device="cpu") -> "SeoulKV4Index": ...

    @classmethod
    def load(cls, path: str | Path) -> "SeoulKV4Index": ...

    def dump(self, path: str | Path) -> None: ...

    def metadata(self) -> dict: ...
```

- `build()`는 CPU BF16 post-RoPE K `[4,T,256]`의 retrieval 구간을 head별로
  클러스터링하고, spill=2 CSR 및 32-centroid 공유 스케일 코드를 만듭니다.
  `build_device`는 PyTorch CPU/CUDA 연산 장치이며 결과는 CPU에 둡니다.
  V는 선택 입력으로 shape, dtype, 장치만 검사합니다. Q는 받지 않습니다.
- `nprobe`는 공통 정수 또는 head별 네 정수입니다. 각 값은 16~512 사이의
  16 배수이며 centroid 수를 넘을 수 없습니다. 실제 후보 수가 런타임의
  Stage-1.5 용량에 맞는지는 초기 실행에서 검사합니다.
- `dump(path)`는 정적 인덱스를 지정한 경로에 NPZ(v4)로 저장합니다.
  K fingerprint는 포함하며, K/V 원본과 Q 및 replay 후보는 저장하지 않습니다.
  `load()`는 이 파일과 기존 grouped Stage-I fixture(v3)를 모두 읽습니다.
  어느 쪽도 재클러스터링하지 않습니다.
- Context, head/cell 순서, sink/local, centroid codec/group scale, CSR,
  per-cell nprobe/lambda와 K fingerprint를 소유합니다. Device 주소는 없습니다.
- Mutable cell dictionary를 public으로 노출하지 않습니다. Runtime별 Q/survivor
  필드는 별도 state에 두어 같은 index를 재사용할 때 이전 실행이 index를 바꾸지 않게 합니다.
- 기존 artifact의 과거 probe Q/coarse survivor 항목은 legacy import 검증 자료일
  뿐입니다. Runtime query로 사용하지 않으며, public index 계약은 정적 summary로
  한정합니다. Loader의 legacy 검증과 runtime용 정적 projection을 분리합니다.
- 생성자에서 전달한 K와 identity/context/순서가 맞지 않으면 실패합니다.
- Sink/local은 index의 retrieval 범위에서 결정됩니다. 다른 값이 필요하면
  해당 범위로 만든 index를 사용합니다. Runtime flag로 구역을 몰래 바꾸지 않습니다.
- append/update API는 아직 없습니다. 새 cache에는 새 인덱스를 만듭니다.

### `SeoulKV4Config`

```python
@dataclass(frozen=True)
class SeoulKV4Config:
    stage15_capacity: int = 13056
    debug: bool = False
```

처음에는 기존 검증 profile을 기본 profile 하나로 제공합니다.

- Top-100, Stage1.5 hardware frontier2048/head, union1600/head,
  Stage1.6 keep400/head, local160/shard, Stage1.5 scorer shard6.
- 위 값 중 하위 fused builder가 다른 값을 거부하는 항목을 자유롭게 조절할 수
  있는 public knob로 노출하지 않습니다. 검증하지 않은 조합을 지원하는 척하지 않습니다.
- `stage15_capacity`만 기존 capacity 검사를 유지하며 필요할 때 늘릴 수 있습니다.
  실제 pool을 capacity에 맞춰 잘라 품질을 낮추는 동작은 하지 않습니다.
- `debug=False`에서는 추가 snapshot/readback을 생성하지 않습니다.
- Per-cell nprobe/lambda 및 sink/local은 index가 소유합니다. Config에 중복 저장하지 않습니다.
- Core topology/CB/semaphore ID/packed layout/느린 이전 구현 선택자는 private입니다.

Device 개폐, trace 영역, L1 분할은 호출자가 관리합니다. 검증된 설정과
필요한 TT-Metal 패치는 [런타임 설정](TT_RUNTIME.md)에 정리했습니다.
라이브러리는 이미 열린 device의 분할을 변경하지 않습니다.

### `SeoulKV4`

```python
class SeoulKV4:
    def __init__(
        self, *, device, k, v, index: SeoulKV4Index,
        config: SeoulKV4Config | None = None,
    ): ...

    def run(self, q: torch.Tensor) -> "DecodeResult": ...
    def synchronize(self) -> None: ...
    def metadata(self) -> dict: ...
    def debug_snapshot(self) -> dict: ...
    def close(self) -> None: ...

    @property
    def initialized(self) -> bool: ...

    # with 구문을 지원하는 __enter__/__exit__도 구현합니다.
```

**Device 소유권:** 전달받은 device를 빌립니다. 라이브러리는 device를 새로
열거나 닫거나 reset하지 않습니다. `close()`는 자기 resource만 정리하며
중복 호출은 안전해야 합니다. 완료를 기다려야 한다면 cleanup에서 명시적으로
동기화합니다. 외부 trace나 raw device tensor handle을 보유했다면 caller가 먼저
해제합니다. 호출자 소유의 TTNN program cache는 비우지 않습니다.

**입력:** 현재 구현의 CPU BF16 K/V/Q부터 지원합니다. Q/K는 이미 RoPE가 적용된
값입니다. 라이브러리는 모델의 projection/QK norm/RoPE/output gate를 실행하지 않습니다.
`run()`에 TTNN Q를 전달하는 API는 첫 버전에서 지원하지 않습니다. Device Q를
몰래 host로 왕복시키는 fallback도 넣지 않습니다.

**초기화:** 생성자는 shape/codec/config 검사와 입력 참조 보관을 수행합니다.
첫 `run(q)`에서 현재처럼 program/workspace 생성과 초기 실행을 수행합니다.
이 cold path에는 compilation/allocation/기존 초기화 readback이 있을 수 있으므로
trace capture나 latency 측정에 사용하지 않습니다. 과거 NPZ의 Q로 초기화하지 않습니다.

**Hot path:** 초기화 이후에는 Q를 한 stable GDDR buffer에 덮어쓰고
`query_prepare_coarse → fine → stage15 → refinement_attention`을 enqueue합니다.
현재와 같은 host Q 전송 비용은 존재합니다. Python/host transport 객체 생성과
device allocation을 구분합니다. Hot path에서 새 device workspace allocation,
CPU cascade reference, 자동 output readback/전체 device synchronize는 하지 않습니다.

**메타데이터/진단:** `metadata()`는 shape, codec, memory/profile, program 목록 등
host에서 이미 아는 정보를 반환하며 장치 readback을 숨기지 않습니다.
`debug_snapshot()`만 명시적 동기화/readback을 허용하며 `debug=True`가 필요합니다.
이 경로는 timed invocation 밖에서 사용합니다. 기본 result에 selected IDs를
얻기 위한 GDDR write를 강제로 추가하지 않습니다.
128K에서 debug buffer까지 두려면 worker L1이 더 필요합니다. 진단 설정은
1,433,728 B/core에서 검증했으며 기본 profile의 partition은 유지했습니다.

**호출 제약:** 같은 instance에 대한 concurrent/reentrant run은 미지원입니다.
지원하지 않는 shape나 상태는 예외를 내고 자동 dense fallback/장치 reset은 하지 않습니다.

### `DecodeResult`

```python
class DecodeResult:
    @property
    def device_tensors(self) -> tuple: ...

    def to_torch(self) -> torch.Tensor: ...
```

- `device_tensors`는 현재 slot별 device output 네 개를 그대로 제공합니다.
  각 native layout은 `[1,32,4,256]`이며 head별 diagonal이 실제 결과입니다.
- `to_torch()`는 동기화 후 device output에서 head별 대각 성분을 읽어
  CPU BF16 `[1,1,16,256]` 텐서를 반환합니다.
- 자동 device-side concat/diagonal gather kernel은 첫 이식에 추가하지 않습니다.
  따라서 모델이 소비할 canonical TT tensor 하나를 반환하는 serving API는 아직 아닙니다.
- Device buffers는 재사용합니다. Result는 다음 run 또는 close 전까지 유효한
  borrowed view이며 과거 output의 독립 snapshot이 아닙니다. 오래 보관하려면
  다음 호출 전에 명시적으로 readback/copy합니다. 반환된 CPU Tensor는 독립적으로 보관합니다.

## 3. Trace와 실행 범위

SeoulKV4는 자체 trace를 만들지 않습니다. 첫 `run(q)`는 프로그램과 작업
공간을 준비하므로 trace 밖에서 호출합니다. 외부에서 trace를 관리하는 경우,
SeoulKV4 인스턴스를 닫기 전에 trace를 해제합니다. Device Q의 직접 바인딩과
외부 모델의 trace 통합은 public API 범위에 포함되지 않습니다.

## 4. 소스 구성

```text
src/seoulkv4/
  __init__.py     # public API, TTNN import는 첫 run에서 수행
  api.py          # device workspace 수명 관리
  config.py       # 실행 설정
  index.py        # 정적 인덱스 생성·저장·로드
  result.py       # device output view와 CPU readback
  _clustering.py  # PyTorch IVF 클러스터링
  _runtime.py     # device program 조립과 실행
  _codecs/        # 키 인코딩과 인덱스 직렬화
  _kernels/       # stage별 TT-Metal 커널 소스 생성
  _compat/        # 검증된 런타임용 TopK-XL 호환 패치
```

밑줄로 시작하는 모듈은 내부 구현입니다. 패키지에는 클러스터링,
인덱스 코덱, host runtime, 커널 소스 생성기가 포함됩니다.
커널의 JIT 컴파일에 필요한 Tenstorrent 헤더와 컴파일러는
[별도로 설정한 런타임](TT_RUNTIME.md)에서 제공합니다.

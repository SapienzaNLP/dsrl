import json
from typing import Generator, Iterator
from classy.data.data_drivers import GenerationDataDriver, GenerationSample, READERS_DICT, GENERATION


class SRLDataDriver(GenerationDataDriver):
    
    def read(self, lines: Iterator[str]) -> Generator[GenerationSample, None, None]:
        for line in lines:
            sample = json.loads(line.strip())
            sample['id'] = int(sample['id'])
            sample['predicate_indices'] = tuple(int(p) for p in sample['predicate_indices'])
            yield GenerationSample(**sample)

    def save(self, samples: Iterator[GenerationSample], path: str):
        with open(path, "w") as f:
            for sample in samples:
                f.write(
                    json.dumps(
                        {
                            "source_sequence": sample.source_sequence,
                            "target_sequence": sample.target_sequence,
                            **sample.get_additional_attributes(),
                        }
                    )
                    + "\n"
                )


READERS_DICT[(GENERATION, 'conll2009')] = SRLDataDriver
READERS_DICT[(GENERATION, 'conll2012')] = SRLDataDriver
READERS_DICT[(GENERATION, 'framenet17')] = SRLDataDriver
READERS_DICT[(GENERATION, 'srl')] = SRLDataDriver

// Slightly updated version of https://github.com/praeclarum/webgpu-torch/
const columnWidths = 30;

function getInputPermutations(benchmarkInputs) {
  function getRemainingPerms(inputs, perms) {
    const nextInputIndex = inputs.length;
    if (nextInputIndex === benchmarkInputs.length) {
      perms.push(inputs);
      return;
    }
    for (let i = 0; i < benchmarkInputs[nextInputIndex].values.length; i++) {
      const inputsCopy = inputs.slice();
      inputsCopy.push(benchmarkInputs[nextInputIndex].values[i]);
      getRemainingPerms(inputsCopy, perms);
    }
  }
  const perms = [];
  getRemainingPerms([], perms);
  return perms;
}

async function runUnaryBenchmarkAsync(benchmark, inputs) {
  const shape = inputs[0];
  const operation = torch[inputs[1]];
  const x = torch.ones(shape);
  // const y = torch.zeros(shape);
  async function runIterationAsync() {
    // console.time('ones');
    // console.timeEnd('ones');
    const start = performance.now();
    let y = x;
    // const y = torch.zeros(shape);
    {
      for (let i = 0; i < benchmark.depth; i++) {
        // operation(x, y);
        y = operation(x);
      }
    }
    // console.log();
    const yar = await y.storage.toTypedArrayAsync(y.dtype);
    // if (yar.length < 1000) {
    //     console.log("y1", yar);
    // }
    // await y.device._device.queue.onSubmittedWorkDone();
    const end = performance.now();
    return (end - start) / benchmark.depth;
  }
  for (let i = 0; i < benchmark.warmupIterations; i++) {
    await runIterationAsync();
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  const ms = [];
  for (let i = 0; i < benchmark.iterations; i++) {
    ms.push(await runIterationAsync());
  }
  return ms;
}

async function runBenchmarkAsync(benchmark, inputs) {
  if (benchmark.type === 'unary') {
    return await runUnaryBenchmarkAsync(benchmark, inputs);
  } else {
    throw `Unknown benchmark type '${benchmark.type}'`;
  }
}

async function loadJsonAsync(url) {
  const response = await fetch(url);
  const o = await response.json();
  return o;
}

async function runBenchmarksAsync() {
  // Load the test JSON
  const benchmarks = await loadJsonAsync('./benchmarks.json');
  const otherResults = await loadJsonAsync('./results.json');

  const otherKeys = Object.keys(otherResults);

  // Create the results string
  let resultsString = '';

  resultsString += padString('Benchmark', columnWidths);
  // Getting adapter info to set in the table
  const adapter = await navigator.gpu?.requestAdapter();
  const adapterDescription = adapter?.info?.description ?? 'Unknown GPU';
  resultsString += padString(adapterDescription, columnWidths);
  for (let ok of otherKeys) {
    resultsString += padString(otherResults[ok].device_name, columnWidths);
  }
  resultsString += '\n';

  const repr = (v) => {
    if (typeof v === 'string') {
      return `'${v}'`;
    } else if (typeof v === 'number') {
      return `${v}`;
    }
    throw `Unknown type '${typeof v}'`;
  };

  // Run the benchmarks
  const benchmarkResults = [];
  for (let b of benchmarks.benchmarks) {
    const inputPerms = getInputPermutations(b.inputs);

    for (let ip of inputPerms) {
      const benchmarkKey = `${b.name}(${ip.map(v => repr(v)).join(', ')})`;
      const result = {key: benchmarkKey, inputs: ip, meanTime: 0.0};
      for (let ok of otherKeys) {
        if (otherResults[ok].results[benchmarkKey]) {
          result[ok] = otherResults[ok].results[benchmarkKey].mean_ms;
        }
      }
      try {
        let times = await runBenchmarkAsync(b, ip);
        const meanTime = times.reduce((a, b) => a + b, 0) / times.length;
        result.meanTime = meanTime;
      } catch (e) {
        result.error = e;
        console.error(e);
      }
      benchmarkResults.push(result);
      appendBenchmarkResult(result);
      await new Promise((resolve) => {
        setTimeout(() => {
          resolve();
        }, 10);
      });
    }
  }

  function padString(str, length) {
    return str.length >= length ? str : str + ' '.repeat(length - str.length);
  }

  function appendBenchmarkResult(b) {
    let row = padString(b.key, columnWidths) +
        padString(b.meanTime.toFixed(3), columnWidths);
    for (let ok of otherKeys) {
      if (otherResults[ok].results[b.key]) {
        const ms = otherResults[ok].results[b.key].mean_ms;
        row += padString(ms.toFixed(5), columnWidths);
      } else {
        row += padString('', columnWidths);
      }
    }
    resultsString += row + '\n';
  }

  return resultsString;
}
